import torch
import cv2
import numpy as np

# For basic type hints
from torch import Tensor
from numpy.typing import NDArray

def remove_infinities(data: Tensor | NDArray, inf_replacement_value = 0.0, in_place = True) -> Tensor | NDArray:
    
    '''
    Helper used to remove +/- inf values, which can sometimes be output
    by the DPT model, especially when using reduced precision dtypes!
    Works on pytorch tensors and numpy arrays
    
    Returns:
        data_with_inf_removed
    '''
    
    try:
        # Works with pytorch tensors
        inf_mask = data.isinf()
        data = data if in_place else data.clone()
        
    except AttributeError:
        # Works with numpy arrays
        inf_mask = np.isinf(data)
        data = data if in_place else data.copy()
    
    # Replace infinity values
    data[inf_mask] = inf_replacement_value
    return data

def normalize_01(data: Tensor | NDArray) -> Tensor | NDArray:
    
    '''
    Helper used to normalize depth prediction, to 0-to-1 range.
    Works on pytorch tensors and numpy arrays
    
    Returns:
        depth_normalized_0_to_1
    '''
    
    pred_min = data.min()
    pred_max = data.max()
    return (data - pred_min) / (pred_max - pred_min)
