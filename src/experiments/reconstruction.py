"""AEROBLADE-style VAE round-trip anomaly scores (single autoencoder; NOT a full AEROBLADE reproduction).

Protocol and score orientation are fixed in outputs/generalization_upgrade/11_reconstruction_detection/PROVENANCE_AND_PROTOCOL.md
before any score is computed. Independent of the classifier pipeline; writes only under 11_reconstruction_detection
(plus master_results.csv rows, and only after output validation passes).

  python -m src.experiments.reconstruction smoke | external | ffpp
"""
import json
import sys
from pathlib import Path

# diffusers/lpips live in an isolated folder so the working environment is never modified; append = never shadows.
_EXTRA = Path(__file__).resolve().parents[2] / ".tier2_pkgs"
if _EXTRA.exists() and str(_EXTRA) not in sys.path:
    sys.path.append(str(_EXTRA))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from skimage.metrics import structural_similarity

from src.eval.calibration import bootstrap_auc
from src.experiments.common import (record_run, save_csv, save_json, atomic_text, update_master, configuration)
from src.experiments.frequency_analysis import sample_spread
from src.experiments.matched_control import components
from src.utils import resolve_device

METRICS = ("mse", "ssim", "lpips")
FOLDER = "11_reconstruction_detection"


def fake_score(metric, values):
    """Fixed, pre-registered orientation: fake score = -error for MSE/LPIPS, +similarity for SSIM."""
    values = np.asarray(values, dtype=float)
    if metric in ("mse", "lpips"):
        return -values
    if metric == "ssim":
        return values
    raise ValueError("Unknown reconstruction score")


@torch.inference_mode()
def reconstruct(vae, rgb):
    """RGB NCHW [0,1] -> posterior mode -> RGB [0,1], direct unscaled AE latent."""
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Finite NCHW RGB [0,1] required")
    latent = vae.encode(rgb * 2 - 1).latent_dist.mode()
    # scaling_factor is for the diffusion UNet; it is not used in this AE-only round trip.
    return (vae.decode(latent).sample / 2 + .5).clamp(0, 1)


def load_models(config, root):
    from diffusers import AutoencoderKL
    from huggingface_hub import HfApi
    import lpips
    settings = config["reconstruction"].copy()
    pin = root / "model_revision.json"
    if pin.exists():
        revision = json.loads(pin.read_text())["revision"]
    else:
        revision = settings.get("revision") or HfApi().model_info(settings["model_id"]).sha
        save_json(pin, {"repository": settings["model_id"], "revision": revision, "license": "creativeml-openrail-m",
                        "external_generator_vae_exact_match": "NOT ESTABLISHED; SD1.5 compatibility assumption (see PROVENANCE_AND_PROTOCOL.md)"})
    device = resolve_device(config["device"])
    vae = AutoencoderKL.from_pretrained(settings["model_id"], subfolder=settings["subfolder"], revision=revision,
                                        torch_dtype=torch.float32, use_safetensors=True).to(device).eval()
    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    settings.update(revision=revision, lpips_network="alex", lpips_variant="standard full-network LPIPS, not paper VGG layer 2")
    return vae, perceptual, device, settings


def read_image(path, size, data_root):
    """Identical, label-blind preprocessing: read -> bicubic resize -> RGB float32 [0,1]."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(data_root) / path
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def score_rows(rows, models, config, settings, keep_examples=4):
    vae, perceptual, device = models
    size, batch = settings["image_size"], settings["batch_size"]
    results, examples, shapes = [], {}, set()
    for start in range(0, len(rows), batch):
        part = rows.iloc[start:start + batch]
        images = [read_image(r.image_path, size, config["data"]["root"]) for r in part.itertuples()]
        inputs = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2)).to(device)
        outputs = reconstruct(vae, inputs)
        with torch.inference_mode():
            perceptual_errors = perceptual(inputs * 2 - 1, outputs * 2 - 1).flatten().cpu().numpy()
        restored = outputs.cpu().numpy().transpose(0, 2, 3, 1)
        shapes.add((tuple(restored.shape[1:]), float(restored.min()) >= 0, float(restored.max()) <= 1))
        for index, (_, row) in enumerate(part.iterrows()):
            image, output = images[index], restored[index]
            item = row.to_dict()
            item.update(mse=float(np.mean((image - output) ** 2)),
                        ssim=float(structural_similarity(image, output, data_range=1, channel_axis=-1)),
                        lpips=float(perceptual_errors[index]))
            results.append(item)
            key = (row.label, row.get("method", ""))
            bucket = examples.setdefault(key, [])
            if len(bucket) < keep_examples:
                bucket.append((image, output, row.get("sample_id", row.image_path)))
        print(f"VAE reconstruction {min(start + batch, len(rows))}/{len(rows)}", flush=True)
    return pd.DataFrame(results), examples, shapes


def example_figure(examples, path, per_group=4):
    items = [(f"label={k[0]} {k[1]}", e) for k, es in sorted(examples.items(), key=lambda kv: str(kv[0])) for e in es[:per_group]]
    fig, axes = plt.subplots(len(items), 3, figsize=(8, 2.7 * len(items)), squeeze=False)
    for axes_row, (group, (image, output, name)) in zip(axes, items):
        for axis, value, title in zip(axes_row, [image, output, np.abs(image - output).mean(axis=-1)],
                                      [group, "VAE reconstruction", "abs RGB error (mean)"]):
            axis.imshow(value, vmin=0, vmax=1, cmap="magma" if value.ndim == 2 else None)
            axis.set_title(title, fontsize=7)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def describe(scores):
    rows = []
    for metric in METRICS:
        for label, frame in scores.groupby("label"):
            v = frame[metric]
            rows.append({"metric": metric, "class": "generated" if label else "real", "n": len(v), "mean": v.mean(), "std": v.std(),
                         "median": v.median(), "q25": v.quantile(.25), "q75": v.quantile(.75), "min": v.min(), "max": v.max()})
    return pd.DataFrame(rows)


def validate(scores, expected_n=None, per_class=None):
    checks = {"finite_all_metrics": bool(np.isfinite(scores[list(METRICS)].to_numpy()).all()),
              "mse_nonnegative": bool((scores.mse >= 0).all()), "ssim_in_range": bool(scores.ssim.between(-1, 1).all()),
              "lpips_nonnegative": bool((scores.lpips >= 0).all()),
              "unique_sample_ids": bool(not scores.sample_id.duplicated().any()),
              "two_classes": set(scores.label) == {0, 1}}
    if expected_n is not None:
        checks["expected_n"] = len(scores) == expected_n
    if per_class is not None:
        checks["balanced"] = bool((scores.label.value_counts() == per_class).all())
    if not all(checks.values()):
        raise ValueError(f"Validation failed: {checks}")
    return checks


def auc_table(scores, config, dataset, groups=None, per_method=False):
    out = []
    for metric in METRICS:
        s = fake_score(metric, scores[metric])
        ci = bootstrap_auc(scores.label, s, groups=groups, replicates=config["bootstrap_replicates"],
                           seed=config["seed"], probability_score=False)
        out.append({"experiment": "reconstruction", "model": f"SD15_VAE_{metric}", "test_dataset": dataset,
                    "protocol_id": config["protocol_id"], "evaluation_unit": "image", "condition": "normalized_256_vae",
                    "auc": float(roc_auc_score(scores.label, s)), **ci, "score_type": "non_probability",
                    "score_orientation": {"mse": "-MSE", "lpips": "-LPIPS", "ssim": "+SSIM"}[metric],
                    "num_samples": len(scores), "seed": config["seed"],
                    "bootstrap_unit": "source_video_component" if groups is not None else "image",
                    "checkpoint": "stable-diffusion-v1-5/stable-diffusion-v1-5:vae (compatibility assumption)"})
        if per_method:
            for method in sorted(set(scores.loc[scores.label == 1, "method"])):
                sub = scores[(scores.label == 0) | (scores.method == method)]
                sm = fake_score(metric, sub[metric])
                out.append({**out[-1], "auc": float(roc_auc_score(sub.label, sm)),
                            **bootstrap_auc(sub.label, sm, groups=None if groups is None else components(sub.source_video_ids),
                                            replicates=config["bootstrap_replicates"], seed=config["seed"], probability_score=False),
                            "test_manipulation": method, "num_samples": len(sub)})
    return out


def distributions(scores, path, title):
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for axis, metric in zip(axes[:3], METRICS):
        for label in (0, 1):
            axis.hist(scores.loc[scores.label == label, metric], bins=30, histtype="step", density=True,
                      label="generated" if label else "real")
        axis.set(title=f"{metric} (raw value)", xlabel="raw reconstruction metric", ylabel="density")
        axis.legend(fontsize=7)
    for metric in METRICS:
        fpr, tpr, _ = roc_curve(scores.label, fake_score(metric, scores[metric]))
        axes[3].plot(fpr, tpr, label=f"{metric}: AUC {roc_auc_score(scores.label, fake_score(metric, scores[metric])):.3f}")
    axes[3].plot([0, 1], [0, 1], "k--", alpha=.4)
    axes[3].set(title="ROC (pre-registered orientation)", xlabel="FPR", ylabel="TPR")
    axes[3].legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def smoke(config):
    root = Path(config["output_dir"]) / FOLDER
    dest = root / "smoke"
    manifest = Path(config["output_dir"]) / "02_external_bias_audit" / "normalized.csv"
    frame = pd.read_csv(manifest)
    n = 8
    rows = pd.concat([frame[frame.label == l].sample(n, random_state=config["seed"]) for l in (0, 1)]).reset_index(drop=True)
    vae, perceptual, device, settings = load_models(config, root)
    record_run(dest, {**config, "reconstruction": settings}, [manifest])
    first, examples, shapes = score_rows(rows, (vae, perceptual, device), config, settings)
    second, _, _ = score_rows(rows, (vae, perceptual, device), config, settings)
    diff = {m: float(np.abs(first[m] - second[m]).max()) for m in METRICS}
    # class-blind preprocessing: the reader is a pure function of the path; verify identical dtype/shape/range for both classes.
    probes = [read_image(r.image_path, settings["image_size"], config["data"]["root"]) for r in rows.itertuples()]
    blind = {(p.dtype.name, p.shape) for p in probes}
    checks = validate(first, expected_n=2 * n, per_class=n)
    report = {"validation": checks, "reconstruction_shape_range_ok": all(s[0] == (256, 256, 3) and s[1] and s[2] for s in shapes),
              "max_abs_difference_between_two_passes": diff, "deterministic": all(v < 1e-6 for v in diff.values()),
              "identical_input_format_for_both_classes": len(blind) == 1, "input_formats": [str(b) for b in blind],
              "vae_revision": settings["revision"], "lpips": settings["lpips_variant"],
              "score_means_by_class_for_information_only": first.groupby("label")[list(METRICS)].mean().to_dict()}
    report["passed"] = bool(report["reconstruction_shape_range_ok"] and report["deterministic"] and report["identical_input_format_for_both_classes"])
    save_csv(dest / "smoke_scores.csv", first)
    example_figure(examples, dest / "smoke_examples.png", per_group=3)
    save_json(dest / "smoke_report.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Smoke test failed; full experiment not run")
    return report


def require_smoke(root):
    path = root / "smoke" / "smoke_report.json"
    if not path.exists() or not json.loads(path.read_text())["passed"]:
        raise SystemExit("Run and pass the smoke test first")


def external(config):
    root = Path(config["output_dir"]) / FOLDER
    require_smoke(root)
    manifest = Path(config["output_dir"]) / "02_external_bias_audit" / "normalized.csv"
    rows = pd.read_csv(manifest)
    models = load_models(config, root)
    record_run(root / "external", {**config, "reconstruction": models[3]}, [manifest])
    scores, examples, _ = score_rows(rows, models[:3], config, models[3])
    checks = validate(scores, expected_n=576, per_class=288)
    save_csv(root / "external" / "reconstruction_scores.csv", scores)
    stats = describe(scores)
    save_csv(root / "external" / "descriptive_statistics.csv", stats)
    metrics = auc_table(scores, config, "external_normalized")
    save_csv(root / "external" / "metrics.csv", pd.DataFrame(metrics))
    distributions(scores, root / "external" / "score_distributions_and_roc.png", "Normalized external cohort (576 images): raw reconstruction metrics; not probabilities")
    example_figure(examples, root / "external" / "reconstruction_examples.png")
    save_json(root / "external" / "validation.json", checks)
    update_master(config["output_dir"], metrics)  # only reached after validate() passed
    print(pd.DataFrame(metrics)[["model", "auc", "auc_ci_lower", "auc_ci_upper", "score_orientation"]].to_string(index=False))
    print(stats.round(5).to_string(index=False))
    return metrics


def ffpp(config):
    root = Path(config["output_dir"]) / FOLDER
    require_smoke(root)
    manifest = Path(config["output_dir"]) / "protocol" / "control" / "test.csv"
    frame = pd.read_csv(manifest, dtype={"source_video_ids": str})
    n = config["reconstruction"]["ff_samples_per_group"]
    rows = pd.concat([sample_spread(f, n, config["seed"]) for _, f in frame.groupby(["label", "method"])]).reset_index(drop=True)
    save_csv(root / "ffpp" / "sampling_manifest.csv", rows)
    models = load_models(config, root)
    record_run(root / "ffpp", {**config, "reconstruction": models[3]}, [manifest])
    scores, examples, _ = score_rows(rows, models[:3], config, models[3])
    checks = validate(scores, expected_n=len(rows))
    scores["cluster_component"] = components(scores.source_video_ids.astype(str))
    save_csv(root / "ffpp" / "reconstruction_scores.csv", scores)
    save_csv(root / "ffpp" / "descriptive_statistics.csv", describe(scores))
    metrics = auc_table(scores, config, "ffpp_source_safe", groups=scores.cluster_component.to_numpy(), per_method=True)
    save_csv(root / "ffpp" / "metrics.csv", pd.DataFrame(metrics))
    distributions(scores, root / "ffpp" / "score_distributions_and_roc.png", "FF++ source-disjoint secondary diagnostic (600 frames)")
    example_figure(examples, root / "ffpp" / "reconstruction_examples.png", per_group=1)
    save_json(root / "ffpp" / "validation.json", checks)
    update_master(config["output_dir"], metrics)
    print(pd.DataFrame(metrics)[["model", "test_manipulation", "auc", "auc_ci_lower", "auc_ci_upper"]].to_string(index=False) if "test_manipulation" in pd.DataFrame(metrics) else "")
    return metrics


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    {"smoke": smoke, "external": external, "ffpp": ffpp}[mode](configuration())
