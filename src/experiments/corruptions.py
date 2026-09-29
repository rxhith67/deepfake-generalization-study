"""Deterministic RGB signal interventions before model-specific normalization."""
import cv2
import numpy as np


def working_image(rgb, size=224):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB uint8 image required')
    return cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA)


def jpeg(rgb,quality,size=224):
    if not isinstance(quality,int) or not 1 <= quality <= 100:
        raise ValueError('JPEG quality must be integer 1..100')
    rgb = working_image(rgb,size)
    ok, encoded = cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,quality])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)


def radial_grid(shape):
    fy,fx = np.meshgrid(np.fft.fftfreq(shape[0]),np.fft.fftfreq(shape[1]),indexing='ij')
    return np.hypot(fx,fy)/.5


def filter_signal(signal,kind,cutoff):
    """Signed linear filtering; radius is cycles/pixel divided by axis Nyquist."""
    if kind not in ('lowpass','highpass') or not 0 < cutoff <= 1:
        raise ValueError('Invalid filter')
    mask = radial_grid(signal.shape[:2]) <= cutoff
    if kind == 'highpass':
        mask = ~mask
    if signal.ndim == 3:
        mask = mask[...,None]
    return np.fft.ifft2(np.fft.fft2(signal,axes=(0,1))*mask,axes=(0,1)).real


def frequency_filter(rgb,kind,cutoff,size=224):
    signal = working_image(rgb,size).astype(np.float64)/255
    result = filter_signal(signal,kind,cutoff)
    if kind == 'highpass':
        result += .5  # Fixed neutral offset; never per-image contrast normalization.
    return np.rint(np.clip(result,0,1)*255).astype(np.uint8)
