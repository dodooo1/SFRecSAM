import numpy as np
import torch
from torch.nn import functional as F
import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from show import *
from itertools import combinations
from scipy.spatial import ConvexHull
import numpy as np
import itertools
import scipy
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
# this folder have all the augmentation functions need for the dataaug

import numpy as np
import random

import matplotlib.pyplot as plt
import cv2
import math
import cv2
import math
# import numpy as np
from bridson import poisson_disc_samples

from bridson import poisson_disc_samples


# the uncertainty-based FNPC block in ratio version
def uc_refine_correct(binary_mask, uncertainty_map, img, box, threshold_uc = 0.9, fn_alpha = 0.0, fn_beta = 1.4,fp_alpha = 0.0, fp_beta = 1.4 ):
    # Average over the channels of img
    img = img * 255
    img_avg = np.mean(img, axis=2)

    # calculate the mean xt(yt)
    mean_value = np.mean(img_avg[(binary_mask[0] > 0)])  # for the placenta task

    # Get the uncertainty threshold value
    uc_threshold = np.min(uncertainty_map)+threshold_uc * (np.max(uncertainty_map) - np.min(uncertainty_map))
    # print(uc_threshold)

    # Identify the U_thre (turn uc to a binary mask)
    U_thre = uncertainty_map > uc_threshold

    # ========== FN correction ==========
    inverse_binary_mask = 1 - binary_mask # get the region where is not covered by pseudo mask

    # get the UH = (1-yt)*U_thre
    FN_UH = U_thre * inverse_binary_mask[0] # DxD

    # get the xUH
    FN_xUH = img_avg * FN_UH # DxD

    # pseudo label FN refine
    # Create a boolean mask where True indicates the condition is met
    FN_condition_mask = (mean_value * fn_alpha < FN_xUH) & (FN_xUH < mean_value * fn_beta)

    # Use numpy's logical_or function to keep the old mask values where the condition is not met
    FN_new_mask = np.logical_or(binary_mask, FN_condition_mask) # the return is DxD since 1xDxD will be consider as DxD in np logical
    # FN_new_mask = np.logical_or(binary_mask, FN_UH)

    # output_mask = np.expand_dims(new_mask, axis=0)
    FN_output_mask = FN_new_mask# 1xDxD

    # ========== FP correction ==========
    FP_UH = U_thre * binary_mask[0]  # DxD

    # get the xUH
    FP_xUH = img_avg * FP_UH  # DxD

    # pseudo label FP refine
    # Create a boolean mask where True indicates the condition is met
    FP_condition_mask = ((mean_value * fp_alpha > FP_xUH) | (FP_xUH > mean_value * fp_beta))&(FP_UH>0)
    # FP_condition_mask =  (FP_xUH > mean_value * fp_beta)

    # Use numpy's logical_or function to keep the old mask values where the condition is not met
    FP_new_mask = np.logical_and(FN_output_mask, np.logical_not(
        FP_condition_mask))  # the return is DxD since 1xDxD will be consider as DxD in np logical
    # FP_new_mask = np.logical_and(FN_output_mask, np.logical_not(FP_UH))

    # output_mask = np.expand_dims(new_mask, axis=0)
    FP_output_mask = FP_new_mask.astype(int)  # 1xDxD
    return FN_output_mask, FN_UH, FN_xUH, FN_condition_mask, FP_output_mask, FP_UH, FP_xUH, FP_condition_mask
    # return FN_output_mask, FN_UH, FP_output_mask, FP_UH