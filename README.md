# SwiftDepth
**SwiftDepth: An Efficient Hybrid CNN-Transformer Model for Self-Supervised Monocular Depth Estimation on Mobile Devices**

2023 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct)

## Results on [KITTI](https://www.cvlibs.net/datasets/kitti/) dataset
| Model            | AbsRel | SqRel | RMSE  | RMSElog | δ < 1.25 | δ < 1.25^2 | δ < 1.25^3 | MParam | GMACs |
| ---------------- | ------ | ----- | ----- | ------- | -------- | ---------- | ---------- | ------ | ----- |
| SwiftDepth-small | 0.110  | 0.830 | 4.700 | 0.187   | 0.882    | 0.962      | 0.982      | 3.6    | 3.6   |
| SwiftDepth       | 0.107  | 0.790 | 4.643 | 0.182   | 0.888    | 0.963      | 0.983      | 6.4    | 4.9   |

Model setup: single frame, monocular training, ResNet18-based pose network, no additional supervision, no post-processing, image resolution of 192x640, and pre-training solely on ImageNet.

## Preparation

Main dependencies are listed in the [requirements.txt](https://github.com/xapaxca/swiftdepth/blob/main/requirements.txt).

Refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) for KITTI dataset preparation.

## Training
**SwiftDepth**
```shell
python train.py --data_path KITTI_DATA_PATH --log_dir LOG_DIR_PATH --model_name SwiftDepth_S_run --split eigen_zhou --num_workers 8 --eval_mono --pose_model_type separate_resnet --learning_rate 1e-4 --num_epochs 20 --scheduler_step_size 17 --config configs.swiftformer_S_pretrained --batch_size 16
```

**SwiftDepth-small**
```shell
python train.py --data_path KITTI_DATA_PATH --log_dir LOG_DIR_PATH --model_name SwiftDepth_XS_run --split eigen_zhou --num_workers 8 --eval_mono --pose_model_type separate_resnet --learning_rate 1e-4 --num_epochs 20 --scheduler_step_size 17 --config configs.swiftformer_XS_pretrained --batch_size 16
```

## Evaluation
**SwiftDepth**
```shell
python evaluate_depth.py --config configs.swiftformer_S_pretrained --load_weights_folder .\weights\swiftdepth --eval_mono --data_path KITTI_DATA_PATH --eval_split eigen --batch_size 10
```

**SwiftDepth-small**
```shell
python evaluate_depth.py --config configs.swiftformer_XS_pretrained --load_weights_folder .\weights\swiftdepth-small --eval_mono --data_path KITTI_DATA_PATH --eval_split eigen --batch_size 10
```

## Acknowledgement
The code is adapted primarily from [Monodepth2](https://github.com/nianticlabs/monodepth2).

It also adapted from [VTDepth](https://github.com/ahbpp/VTDepth), [EPCDepth](https://github.com/prstrive/EPCDepth), and [Lite-Mono](https://github.com/noahzn/Lite-Mono).

Special thanks to [SwiftFormer](https://github.com/Amshaker/SwiftFormer) for publicly available code and weights for their efficient backbone used as an encoder for SwiftDepth.

## Citation
```
@InProceedings{10322131,
  author={Luginov, Albert and Makarov, Ilya},
  booktitle={2023 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct)}, 
  title={SwiftDepth: An Efficient Hybrid CNN-Transformer Model for Self-Supervised Monocular Depth Estimation on Mobile Devices},
  year={2023},
  pages={642-647},
  doi={10.1109/ISMAR-Adjunct60411.2023.00137}
}
```
