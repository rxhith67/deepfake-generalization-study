import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "download_deepfakeface_subset.py"
SPEC = importlib.util.spec_from_file_location("download_deepfakeface_subset", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_choose_pairs_is_matched_and_deterministic():
    real = ["wiki/a/1.jpg", "wiki/a/2.jpg", "wiki/a/real-only.jpg", "wiki/"]
    fake = ["text2img/a/1.jpg", "text2img/a/2.jpg", "text2img/a/fake-only.jpg"]
    assert module.choose_pairs(real, fake, 2, 42) == module.choose_pairs(real, fake, 2, 42)
    assert set(module.choose_pairs(real, fake, 2, 42)) == {"a/1.jpg", "a/2.jpg"}


def test_choose_pairs_spreads_selection_across_archive_order():
    real = [f"wiki/{index}.jpg" for index in range(100)]
    fake = [f"text2img/{index}.jpg" for index in range(100)]
    chosen = [int(Path(name).stem) for name in module.choose_pairs(real, fake, 20, 7)]
    assert len(chosen) == 20
    assert min(chosen) < 10 and max(chosen) >= 90
