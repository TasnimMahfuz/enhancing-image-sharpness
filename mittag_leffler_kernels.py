import numpy as np
from scipy.special import gamma


def mittag_leffler_standard(lambda_val, delta, theta, terms = 100):
    result = 0

    for k in range(terms):
        
        term = (lambda_val ** k)/gamma(theta + delta*k)
        result += term

        if abs(term) < 1e-15:
            break
        
    return result 

def mittag_leffler_paper(lambda_val, delta, theta, terms = 300):
    result = 0

    for k in range(2, terms):
        
        term = (lambda_val**k)/gamma(theta + delta*k)
        result += term

        if abs(term) < 1e-15:
            break

    return result

def calculate_coefficients(delta = 1, lambda_val = 0.01, theta = 0.1, terms = 300):
    
    E_delta_theta = mittag_leffler_paper(lambda_val, delta, theta, terms)

    o1 = 1.0
    o2 = (gamma(1+ theta)*E_delta_theta)/75.005
    o3 = (gamma(2 + theta)*E_delta_theta)/1.125075


    return o1, o2, o3


def create_masks(o1,o2, o3):
    mask_0 = np.array([
        [0,0,0],
        [o1,o2, o3],
        [0,0,0]
    ])

    mask_90 = np.array([
        [0,o3,0],
        [0,o2,0],
        [0, o1, 0]
    ])

    mask_45 = np.array([
        [0,0,o3],
        [0,o2,0],
        [o1,0,0]
    ])

    mask_135 = np.array([
        [o3,0,0],
        [0,o2,0],
        [0,0,o1]
    ])

    return mask_0, mask_90, mask_45, mask_135


