# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

### MODIFIED

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
