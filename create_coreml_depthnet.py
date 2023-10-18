from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import coremltools as ct
import argparse
import importlib
import os


model_to_config = {
    "SwiftDepth": "configs.swiftformer_S_pretrained",
    "SwiftDepth-small": "configs.swiftformer_XS_pretrained",
}

model_names = model_to_config.keys()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Scripts to create CoreML model')

    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=model_names, required=True)
    parser.add_argument('--save_name', type=str,
                        help='name of a CoreML model to save', required=True)
    parser.add_argument("--models_path",
                        type=str,
                        help="path to the training data", required=True)

    return parser.parse_args()

def load_weights(model, path):
    try:
        ckpt_path = path # weights pretrained on imagenet
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt
        state_dict = _state_dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
        print(f"The following pretrained weights have been initialized: {ckpt_path}")
        print(f"{missing_keys=}")
        print(f"{unexpected_keys=}")
        print(f"{len(state_dict.keys())=}")
        print(f"{len(unexpected_keys)=}")
    except:
        print("No pretrained weights are initialized")
    return model, state_dict

class DepthNet(nn.Module):
    def __init__(self, encoder, depth):
        super().__init__()
        self.encoder = encoder
        self.depth = depth

    def disp_to_depth(self, disp, min_depth=0.1, max_depth=100):
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def forward(self, img):
        feat = self.encoder(img)
        disps = self.depth(feat)
        _, depth = self.disp_to_depth(disps[("disp", 0)])
        return depth
    
def get_config(config_path):
    config = importlib.import_module(config_path)
    return config

def create_coreml_model(args):
    assert args.model_name is not None
    assert args.save_name is not None
    assert args.models_path is not None
    config = get_config(model_to_config[args.model_name])
    encoder, depth = config.build_depth(pretrained=False)
    encoder_path = os.path.join(args.models_path, "encoder.pth")
    depth_path = os.path.join(args.models_path, "depth.pth")
    encoder, encoder_state_dict = load_weights(encoder, path=encoder_path)
    depth, _ = load_weights(depth, path=depth_path)
    depth_net = DepthNet(encoder=encoder, depth=depth)
    depth_net.eval()

    height = encoder_state_dict['height']
    width = encoder_state_dict['width']

    example_input = torch.rand(1, 3, height, width) 
    traced_model = torch.jit.trace(depth_net, example_input)

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=example_input.shape)]
    )

    save_folder = os.path.join(args.models_path, "coreml")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_name = os.path.join(save_folder, f"{args.save_name}.mlpackage")
    model.save(file_name)

if __name__ == "__main__":
    args = parse_args()
    create_coreml_model(args)
