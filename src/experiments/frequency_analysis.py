"""Method-balanced FFT evidence with matched-domain real references."""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.common import configuration,save_csv,save_json,record_run,atomic_text
from src.experiments.corruptions import radial_grid


def sample_spread(frame,n,seed):
    """Round-robin videos after seeded shuffling; avoids selecting only adjacent frames."""
    frame = frame.sample(frac=1,random_state=seed).copy()
    groups = 'actual_video_id' if 'actual_video_id' in frame else 'video_id'
    frame['_rank'] = frame.groupby(groups).cumcount()
    return frame.sort_values('_rank',kind='stable').head(n).drop(columns='_rank')


def run(config):
    root = Path(config['output_dir'])
    dest = root/'06_frequency_analysis'
    ff = pd.read_csv(root/'protocol/control/test.csv')
    ext = pd.read_csv(root/'02_external_bias_audit/normalized.csv')
    record_run(dest,config,[root/'protocol/control/test.csv',root/'02_external_bias_audit/normalized.csv'])
    groups = {f'FF++ {method}':g for method,g in ff.groupby('method')}
    groups.update({f'External {"fake" if label else "real"}':g for label,g in ext.groupby('label')})
    quota = min(config['frequency']['samples_per_group'],min(map(len,groups.values())))
    spectra, summaries, selections, radial = {}, [], [], []
    radius = radial_grid((224,224))
    radial_edges = np.linspace(0,np.sqrt(2),51)
    for name,frame in groups.items():
        selected = sample_spread(frame,quota,config['seed'])
        selections.append(selected.assign(analysis_group=name))
        logs, powers = [], []
        for row in selected.itertuples():
            path = Path(row.image_path)
            if not path.is_absolute():
                path = Path(config['data']['root'])/path
            rgb = cv2.imread(str(path))
            if rgb is None:
                raise FileNotFoundError(path)
            gray = cv2.cvtColor(cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY).astype(float)/255
            transform = np.fft.fft2(gray-gray.mean())
            logs.append(np.fft.fftshift(np.log1p(np.abs(transform))))
            power = np.abs(transform)**2
            power[0,0] = 0
            powers.append(power)
            total = max(power.sum(),1e-12)
            summaries.append({'group':name,'image_path':str(path),'low_energy_ratio':power[(radius>0)&(radius<=.1)].sum()/total,
                              'mid_energy_ratio':power[(radius>.1)&(radius<=.3)].sum()/total,
                              'high_energy_ratio':power[radius>.3].sum()/total})
        spectra[name] = np.mean(logs,axis=0)
        mean_power = np.mean(powers,axis=0)
        for lo,hi in zip(radial_edges[:-1],radial_edges[1:]):
            mask = (radius>lo)&(radius<=hi)
            radial.append({'group':name,'radius':(lo+hi)/2,'mean_power':float(mean_power[mask].mean()) if mask.any() else 0})
    save_csv(dest/'sampling_manifest.csv',pd.concat(selections,ignore_index=True))
    save_csv(dest/'spectral_energy_per_image.csv',pd.DataFrame(summaries))
    save_csv(dest/'spectral_energy_summary.csv',pd.DataFrame(summaries).groupby('group').mean(numeric_only=True).reset_index())
    save_csv(dest/'radial_power.csv',pd.DataFrame(radial))
    fake_groups = [n for n in groups if n not in ('FF++ real','External real')]
    fig,axes = plt.subplots(len(fake_groups),3,figsize=(10,3*len(fake_groups)))
    vmax = max(s.max() for s in spectra.values())
    diffs = [spectra[n]-spectra['External real' if n.startswith('External') else 'FF++ real'] for n in fake_groups]
    scale = max(np.abs(d).max() for d in diffs)
    for axes_row,name,diff in zip(axes,fake_groups,diffs):
        ref = 'External real' if name.startswith('External') else 'FF++ real'
        for axis,im,title in zip(axes_row,[spectra[ref],spectra[name],diff],[ref,name,'Fake - real']):
            difference = title == 'Fake - real'
            axis.imshow(im,cmap='coolwarm' if difference else 'magma',vmin=-scale if difference else 0,vmax=scale if difference else vmax)
            axis.set_title(title)
            axis.axis('off')
    fig.tight_layout()
    fig.savefig(dest/'mean_spectra_and_differences.png',dpi=250)
    plt.close(fig)
    fig,axis = plt.subplots(figsize=(8,5))
    for name,frame in pd.DataFrame(radial).groupby('group'):
        axis.semilogy(frame.radius,frame.mean_power.clip(lower=1e-12),label=name)
    axis.set(xlabel='Radius / axis Nyquist',ylabel='Mean power',title='Mean-subtracted grayscale; no window')
    axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(dest/'radial_power.png',dpi=250); plt.close(fig)
    save_json(dest/'parameters.json',{'samples_per_group':quota,'gray':'OpenCV BT.601','mean_subtract':True,'window':None,
                                     'dc_excluded':True,'band_edges':[.1,.3],'radius':'axis_nyquist'})
    atomic_text(dest/'frequency_analysis.md','# Frequency evidence\n\nFixed method-balanced, video-spread sampling. '
                'FF++ manipulations use FF++ real references; diffusion uses normalized external real. '
                'Mean-subtracted grayscale FFT, no window, DC excluded from energy; fixed bands 0-.1, .1-.3, >.3. '
                'Patterns are descriptive, not proof of universal artifacts or causality.\n')


if __name__ == '__main__':
    run(configuration())
