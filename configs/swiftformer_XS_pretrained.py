'''
The code is adapted/adopted primarily from Monodepth2:
GitHub: https://github.com/nianticlabs/monodepth2
Paper: https://arxiv.org/abs/1806.01260

It also adapted from VTDepth:
GitHub: https://github.com/ahbpp/VTDepth
Paper: https://ieeexplore.ieee.org/document/9995672
'''

import networks
import torch
import torch.optim as optim
from networks.swiftformer import SwiftFormer_XS


def load_weights(encoder):
    try:
        ckpt_path = "encoder_pretrained_weights\SwiftFormer_XS_ckpt.pth" # weights pretrained on imagenet
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt
        state_dict = _state_dict
        missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, False)
        print(f"The following ImageNet pretrained weights were loaded: {ckpt_path}")
        print(f"{missing_keys=}")
        print(f"{unexpected_keys=}")
        print(f"{len(state_dict.keys())=}")
        print(f"{len(unexpected_keys)=}")
    except:
        print("No weights, pretrained on ImageNet")
    return encoder

def build_depth(pretrained=True):
    encoder = SwiftFormer_XS()
    if pretrained:
        encoder = load_weights(encoder)
    decoder = networks.DepthDecoder(num_ch_enc=encoder.embed_dims, enc_name="SwiftFormer_XS")
    return encoder, decoder

def build_models(opt, device):
    models = {}
    depth_encoder, depth_decoder = build_depth(pretrained=True)
    models["encoder"] = depth_encoder.to(device)
    models["depth"] = depth_decoder.to(device)
    num_input_frames = len(opt.frame_ids)
    num_pose_frames = 2
    if opt.pose_model_type == "separate_resnet":
        models["pose_encoder"] = networks.ResnetEncoder(
            opt.num_layers,
            opt.weights_init == "pretrained",
            num_input_images=num_pose_frames).to(device)

        models["pose"] = networks.PoseDecoder(
            models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2).to(device)

    elif opt.pose_model_type == "shared":
        models["pose"] = networks.PoseDecoder(
            models["encoder"].num_ch_enc, num_pose_frames).to(device)

    elif opt.pose_model_type == "posecnn":
        models["pose"] = networks.PoseCNN(
            num_input_frames if opt.pose_model_input == "all" else 2).to(device)
    return models

def build_optim(models, opt):
    parameters = []
    parameters.append({'params': models["encoder"].parameters()})
    parameters.append({'params': models["depth"].parameters()})
    parameters.append({'params': models["pose_encoder"].parameters()})
    parameters.append({'params': models["pose"].parameters()})
    model_optimizer = optim.Adam(parameters, opt.learning_rate)
    return model_optimizer

def build_scheduler(model_optimizer, opt):
    model_lr_scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[15, 17, 19], gamma=0.3)
    return model_lr_scheduler


