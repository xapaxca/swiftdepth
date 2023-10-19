# The code is adapted from Monodepth2, VTDepth, and Lite-Mono:
# Monodepth2 GitHub: https://github.com/nianticlabs/monodepth2
# Monodepth2 paper: https://arxiv.org/abs/1806.01260
# VTDepth GitHub: https://github.com/ahbpp/VTDepth
# VTDepth paper: https://ieeexplore.ieee.org/document/9995672
# Lite-Mono GitHub: https://github.com/noahzn/Lite-Mono
# Lite-Mono paper: https://arxiv.org/abs/2211.13202

from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import importlib
from thop import clever_format
from thop import profile

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def profile_once(encoder, decoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    x_d = encoder(x_e)
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d

def load_weights(model, path, model_name="unknown"):
    try:
        ckpt_path = path
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt
        state_dict = _state_dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
        print(f"The following pretrained weights have been loaded for {model_name}: {ckpt_path}")
    except:
        print(f"No pretrained weights are initialized for {model_name}")
    return model

def save_numpy(data, folder, name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_path = os.path.join(folder, name)
    np.save(output_path, data)
    print(f"-> Saving predicted {name} to {output_path}")

def get_config(opt):
    config = importlib.import_module(opt.config)
    return config

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    config = get_config(opt)
    results_save_dir = os.path.join(opt.load_weights_folder, "{}_split".format(opt.eval_split))
    inputs_colors = []

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        img_ext = '.png' if opt.png else '.jpg'

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                    opt.height, opt.width,
                                    [0], 4, is_train=False, img_ext=img_ext)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder, depth_decoder = config.build_depth(pretrained=False)

        encoder = load_weights(encoder, encoder_path, "encoder")
        depth_decoder = load_weights(depth_decoder, decoder_path, "decoder")

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        encoder_params = sum([p.numel() for p in encoder.parameters()]) / 10 ** 6
        decoder_params = sum([p.numel() for p in depth_decoder.parameters()]) / 10 ** 6
        print("Encoder params: {:3f} Decoder params: {:3f}".format(encoder_params, decoder_params))

        encoder_dict = torch.load(encoder_path)

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        
        isProfile = True

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()
                input_batch_numpy = np.split(input_color.cpu().numpy(), input_color.size(0))
                input_batch_numpy = [inp.squeeze(0) for inp in input_batch_numpy]
                inputs_colors.extend(input_batch_numpy)

                if isProfile:
                    flops, params, flops_e, params_e, flops_d, params_d = profile_once(encoder, depth_decoder, input_color)
                    isProfile = False

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        save_numpy(pred_disps, results_save_dir, "disps.npy")

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    unmasked_pred_depths = []
    unmasked_gt_depths = []
    masks = []
    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0
        masks.append(mask)
        unmasked_pred_depth = pred_depth.copy()
        unmasked_gt_depth = gt_depth.copy()
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
            unmasked_pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

        unmasked_pred_depth[unmasked_pred_depth < MIN_DEPTH] = MIN_DEPTH
        unmasked_pred_depth[unmasked_pred_depth > MAX_DEPTH] = MAX_DEPTH

        unmasked_pred_depths.append(unmasked_pred_depth)
        unmasked_gt_depths.append(unmasked_gt_depth)

    every_n = 70
    save_numpy(np.array(masks[::every_n], dtype=object), results_save_dir, "masks.npy")
    save_numpy(np.array(unmasked_pred_depths[::every_n], dtype=object), results_save_dir, "pred_depths.npy")
    save_numpy(np.array(unmasked_gt_depths[::every_n], dtype=object), results_save_dir, "gt_depths.npy")
    save_numpy(np.array(errors[::every_n]), results_save_dir, "errors.npy")

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        save_numpy(med, results_save_dir, "median.npy")
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    save_numpy(mean_errors, results_save_dir, "mean_errors.npy")

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n  " + ("flops_total: {0}, params_total: {1}\n  flops_encoder: {2}, params_encoder: {3}\n  flops_decoder: {4}, params_decoder: {5}").format(flops, params, flops_e, params_e, flops_d, params_d))
    print("\n-> Done!")
    return inputs_colors, masks, unmasked_pred_depths, unmasked_gt_depths, errors

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
