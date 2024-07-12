import os.path

import numpy as np
import random
import torch
import math

from prompt_aug import overlay_uncertainty_on_image
from uncertainty_refine import *
import matplotlib as mpl


def generate_box(ref_mask):
    H, W = ref_mask.shape[1], ref_mask.shape[2]
    y, x = np.nonzero(ref_mask[0, :, :])
    if y.size == 0:  # in case some input doesn't have mask
        x_min = 128 - 20
        x_max = 128 + 20
        y_min = 128 - 20
        y_max = 128 + 20
        point_label = 0
    else:
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        point_label = 1
    w = (x_max - x_min) * 1.3
    h = (y_max - y_min) * 1.3
    x1 = int((x_min + x_max) / 2 - w / 2)
    if x1 < 0:
        x1 = 0
    y1 = int((y_min + y_max) / 2 - h / 2)
    if y1 < 0:
        y1 = 0
    x2 = int((x_min + x_max) / 2 + w / 2)
    if x2 > W:
        x2 = W
    y2 = int((y_min + y_max) / 2 + h / 2)
    if y2 > H:
        y2 = H
    return np.array([x1, y1, x2, y2])


# generate new point
def sample_points(ref_mask, pt, M, N):
    y, x = np.nonzero(ref_mask[:, :])
    if y.size == 0:  # in case some input doesn't have mask
        x_min = 128 - 20
        x_max = 128 + 20
        y_min = 128 - 20
        y_max = 128 + 20
    else:
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()

    # Calculate side lengths
    ES = min(x_max - x_min, y_max - y_min)

    # Calculate the radius
    radius = ES / M

    points = []
    points_label = []

    # Function to sample points from a circle
    def sample_positive_points_from_circle(center, radius, num_points):
        for _ in range(num_points):
            # Sample a point in polar coordinates and convert it to cartesian
            theta = 2 * np.pi * random.random()
            r = radius * np.sqrt(random.random())
            xp = center[0] + r * np.cos(theta)
            yp = center[1] + r * np.sin(theta)
            points.append([[xp, yp]])
            points_label.append([1])

    sample_positive_points_from_circle(pt, radius, N)

    return radius, points, points_label


def calculate_aleatoric_uncertainty(mask_list):
    # stack the mask samples in the mask_list to calculate the frequency of each pixel
    all_masks = np.vstack(mask_list)

    # calculate the frequency of 1 for each pixel location
    frequency = np.mean(all_masks, axis=0)

    # calculate the aleatoric uncertainty
    aleatoric_uncertainty = -0.5 * (
            frequency * np.log(frequency + 10e-7) + (1 - frequency) * np.log((1 - frequency) + 10e-7))

    return aleatoric_uncertainty


def predict_ave_mask(new_point_list, new_point_label_list, model, image, hfc):
    # generate mask for point_list
    mask_list = []
    for i in range(0, len(new_point_list)):
        pt_pos = np.reshape(new_point_list[i], (1, 1, 2))
        tensor1 = torch.tensor(pt_pos).cuda()
        pt_label = np.expand_dims(new_point_label_list[i], axis=0)
        tensor2 = torch.tensor(pt_label).cuda()
        pt = (tensor1, tensor2)

        pred = model(image, hfc, pt)
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        mask_list.append(seg)

    ave_mask = np.mean(mask_list, axis=0)
    ave_thre = 0.5
    ave_mask = np.where(ave_mask > ave_thre * (np.max(ave_mask)), 1, 0)

    return ave_mask


def predict_dfc_mask(new_point_list, new_point_label_list, input_box, model, image, hfc, original_mask, gt_mask,
                      image_name, output_path):
    mask_list = []
    for i in range(0, len(new_point_list)):
        pt_pos = np.reshape(new_point_list[i], (1, 1, 2))
        tensor1 = torch.tensor(pt_pos).cuda()
        pt_label = np.expand_dims(new_point_label_list[i], axis=0)
        tensor2 = torch.tensor(pt_label).cuda()
        pt = (tensor1, tensor2)

        pred = model(image, hfc, pt)
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        mask_list.append(seg)

    input_mask_list = mask_list.copy()
    aleatoric_uncertainty_map = calculate_aleatoric_uncertainty(input_mask_list)  # 128x128
    original_uncertainty_map = aleatoric_uncertainty_map.copy()
    ave_mask = np.mean(mask_list, axis=0)
    # ave_mask = np.where(ave_mask > 0.5, 1, 0)
    ave_thre = 0.5
    ave_mask = np.where(ave_mask > ave_thre * (np.max(ave_mask)), 1, 0)

    image = image[0, :, :, :].detach().cpu().numpy()
    image = image.transpose(1, 2, 0)
    FN_final_mask, FN_UH, FN_xUH, FN_condition_mask, FP_final_mask, FP_UH, FP_xUH, FP_condition_mask = uc_refine_correct(
        ave_mask.copy(),
        aleatoric_uncertainty_map.copy(),
        image.copy(),
        input_box)

    # visualization the DFC process

    fig, axs = plt.subplots(3, 4, figsize=(10 * 3, 10 * 4))

    axs[0, 0].imshow(image)
    original_mask[original_mask[:, :] == 255] = 1
    show_mask(original_mask, axs[0, 0])
    # show_points(new_point_list)
    # axs[0, 0].set_title(f'Ori_D_{case_dice_original:.2f}_A_{case_assd_original:.2f}_H_{case_hf_original:.2f}', fontsize=18)
    axs[0, 0].set_title(f'Ori', fontsize=18)

    axs[0, 1].imshow(image)
    show_mask(ave_mask.transpose(1, 2, 0), axs[0, 1])
    axs[0, 1].set_title(f'Ave', fontsize=18)

    #
    vis_img = image.copy()
    # axs[4].imshow(test_image)
    # show_uncertainty(aleatoric_uncertainty_map, plt.gca())

    overlay_uncertainty_on_image(vis_img, original_uncertainty_map, axs[0, 2])
    axs[0, 2].set_title(f"Uncertainty Map", fontsize=18)

    axs[0, 3].imshow(image)
    show_mask(gt_mask.transpose(1, 2, 0), axs[0, 3])
    axs[0, 3].set_title(f'GT', fontsize=18)

    # FN UH
    axs[1, 0].imshow(FN_UH)
    axs[1, 0].set_title(f"FN UH", fontsize=18)

    # # FN xUH
    axs[1, 1].imshow(FN_xUH, cmap='gray', vmin=0, vmax=255)
    axs[1, 1].set_title(f"FN xUH", fontsize=18)

    # # FN condition
    axs[1, 2].imshow(FN_condition_mask)
    axs[1, 2].set_title(f"FN_condition_mask", fontsize=18)

    axs[1, 3].imshow(image)
    show_mask(FN_final_mask.transpose(1, 2, 0), axs[1, 3])
    axs[1, 3].set_title(f'FN', fontsize=18)

    # FP UH
    axs[2, 0].imshow(FP_UH)
    axs[2, 0].set_title(f"FP UH", fontsize=18)

    # # FP xUH
    axs[2, 1].imshow(FP_xUH, cmap='gray', vmin=0, vmax=255)
    axs[2, 1].set_title(f"FP xUH", fontsize=18)

    # # FN condition
    axs[2, 2].imshow(FP_condition_mask)
    axs[2, 2].set_title(f"FP_condition_mask", fontsize=18)

    axs[2, 3].imshow(image)
    show_mask(FP_final_mask.transpose(1, 2, 0), axs[2, 3])
    axs[2, 3].set_title(f'FP_FN', fontsize=18)

    # Save the subplot panel as a single image
    vis_mask_output_path = os.path.join(output_path, "UC", f'Results_Comparison_{image_name}.png')
    plt.savefig(vis_mask_output_path, format='png')

    return FN_final_mask, FP_final_mask
