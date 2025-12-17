import numpy as np
from skimage.metrics import structural_similarity as ssim

def mse(original, enhanced):
    original = original.astype(np.float64)
    enhanced = enhanced.astype(np.float64)

    return np.mean((original - enhanced) ** 2)


def rmse(original, enhanced):
    return np.sqrt(mse(original, enhanced))


def mae(original, enhanced):
    original = original.astype(np.float64)
    enhanced = enhanced.astype(np.float64)

    return np.mean(np.abs(original - enhanced))

