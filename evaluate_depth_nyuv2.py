'''
The code is adapted/adopted primarily from Monodepth2:
GitHub: https://github.com/nianticlabs/monodepth2
Paper: https://arxiv.org/abs/1806.01260

It also adapted from EPCDepth:
GitHub: https://github.com/prstrive/EPCDepth
Paper: https://arxiv.org/abs/2109.12484
'''

from __future__ import absolute_import, division, print_function

import torch
from torch.utils.data import DataLoader
import numpy as np
import progressbar
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image
import os
import argparse
import importlib

from nyuv2.nyu_dataset import NYUTestDataset

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def compute_multi_errors(gt, pred):
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

    depth_errors = np.array([abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])

    return depth_errors

def post_process_disparity(l_disp, r_disp):
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


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

def get_config(config):
    config = importlib.import_module(config)
    return config

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="path to the data")
parser.add_argument("--save_dir", type=str, help="path to save results", metavar="PATH")
parser.add_argument("--img_height", type=int, help="input image height", default=192)
parser.add_argument("--img_width", type=int, help="input image width", default=256)
parser.add_argument("--min_depth", type=float, help="minimum depth", default=0.1)
parser.add_argument("--max_depth", type=float, help="maximum depth", default=10.0)
parser.add_argument("--post_process", action="store_true")
parser.add_argument("--config", help="Path to config file", required=True, type=str)
parser.add_argument("--load_weights_folder", type=str, help="name of model to load")
parser.add_argument("--save_name", type=str, help="name of model to save")


args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    args.load_weights_folder = os.path.expanduser(args.load_weights_folder)

    assert os.path.isdir(args.load_weights_folder), \
        "Cannot find a folder at {}".format(args.load_weights_folder)

    print("-> Loading weights from {}".format(args.load_weights_folder))

    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    config = get_config(args.config)
    
    encoder, depth_decoder = config.build_depth(pretrained=False)

    encoder = load_weights(encoder, encoder_path, "encoder").to(device)
    depth_decoder = load_weights(depth_decoder, decoder_path, "decoder").to(device)

    val_dataset = NYUTestDataset(data_path=args.data_path, height=args.img_height, width=args.img_width)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    encoder.eval()
    depth_decoder.eval()

    pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ", progressbar.Bar(), " ",
                progressbar.Timer(), ",", progressbar.Variable('abs_rel', width=1), ",", progressbar.Variable('sq_rel', width=1), ",",
                progressbar.Variable('rmse', width=1, precision=4), ",", progressbar.Variable('rmse_log', width=1), ",",
                progressbar.Variable('a1', width=1), ",", progressbar.Variable('a2', width=1), ",", progressbar.Variable('a3', width=1)]
    pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(val_loader), prefix="Val:").start()

    depth_errors_meter = AverageMeter()
    for batch, data in enumerate(val_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device, non_blocking=True)

        ipt = data[0]
        if args.post_process:
            # Post-processed results require each image to have two forward passes
            ipt = torch.cat((ipt, torch.flip(ipt, [3])), 0)

        pred_disps = depth_decoder(encoder(ipt))
        pred_disps, _ = disp_to_depth(pred_disps[("disp", 0)], args.min_depth, args.max_depth)
        pred_disps = pred_disps.data.cpu()[:, 0].numpy()  # (b, h, w)

        if args.post_process:
            N = pred_disps.shape[0] // 2
            pred_disps = post_process_disparity(pred_disps[:N], pred_disps[N:, :, ::-1])

        pred_disp = pred_disps[0]
        vmax = np.percentile(pred_disp, 95)
        normalizer = mpl.colors.Normalize(vmin=pred_disp.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(pred_disp)[:, :, :3] * 255).astype(np.uint8)
        im = Image.fromarray(colormapped_im)
        os.makedirs(args.save_dir, exist_ok=True)

        save_folder_name = f"{args.save_name}-{args.img_height}-{args.img_width}"
        os.makedirs(os.path.join(args.save_dir, save_folder_name), exist_ok=True)

        im.save(os.path.join(args.save_dir, save_folder_name, "disp{}.png".format(batch)))

        pred_disp = 1 / data[1][0, 0].data.cpu().numpy()

        gt_depth = data[1][0, 0].data.cpu().numpy()  # (h2, w2)
        pred_depth = 1 / pred_disps[0]  # (h, w)
        gt_height, gt_width = gt_depth.shape[:2]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

        mask = gt_depth > 0
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        ratio = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= ratio

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        depth_errors = compute_multi_errors(gt_depth, pred_depth)

        depth_errors_meter.update(depth_errors, 1)

        pbar.update(batch, abs_rel=depth_errors_meter.avg[0],
                    sq_rel=depth_errors_meter.avg[1],
                    rmse=depth_errors_meter.avg[2],
                    rmse_log=depth_errors_meter.avg[3],
                    a1=depth_errors_meter.avg[4],
                    a2=depth_errors_meter.avg[5],
                    a3=depth_errors_meter.avg[6])

    pbar.finish()
