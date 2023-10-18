'''
The code is adapted/adopted primarily from Monodepth2:
GitHub: https://github.com/nianticlabs/monodepth2
Paper: https://arxiv.org/abs/1806.01260

It also adapted from VTDepth:
GitHub: https://github.com/ahbpp/VTDepth
Paper: https://ieeexplore.ieee.org/document/9995672
'''

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
import importlib

options = MonodepthOptions()
opts = options.parse()


def get_config(opt):
     config = importlib.import_module(opt.config)
     return config

if __name__ == "__main__":
    config = get_config(opts)
    trainer = Trainer(opts, config)
    trainer.train()
