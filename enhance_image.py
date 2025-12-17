from mittag_leffler_kernels import mittag_leffler_paper, calculate_coefficients, create_masks
from skimage import color
from scipy.ndimage import convolve
import numpy as np

def enhance_image(image, theta = 0.1, delta = 1, lambda_val = 0.01, normalize= False, return_all_directions=False):
    if image.ndim == 3:
        image_gray = color.rgb2gray(image)
    else:
        image_gray = image.astype(float)


    if normalize:
        image_gray = (image_gray - image_gray.min())/(image_gray.max() - image_gray.min())


    o1, o2, o3 = calculate_coefficients(delta=delta, lambda_val=lambda_val, theta = theta)

    mask_0, mask_90, mask_45, mask_135 = create_masks(o1,o2, o3)

    enhanced_0 = convolve(image_gray, mask_0, mode="reflect")
    enhanced_90 = convolve(image_gray, mask_90, mode = "reflect")
    enhanced_45 = convolve(image_gray, mask_45, mode = "reflect")
    enhanced_135 = convolve(image_gray, mask_135, mode = "reflect")

    enhanced_avg = (enhanced_0 +  enhanced_90 + enhanced_45 + enhanced_135)/4.0

    if return_all_directions:
        return (image_gray, enhanced_avg, enhanced_0, enhanced_90, enhanced_45, enhanced_135)
    return (image_gray, enhanced_avg)
