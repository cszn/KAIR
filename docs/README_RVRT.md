# [Recurrent Video Restoration Transformer with Guided Deformable Attention (RVRT, NeurlPS2022)](https://github.com/JingyunLiang/RVRT)
[arxiv](https://arxiv.org/abs/2206.02146)
**|** 
[supplementary](https://github.com/JingyunLiang/RVRT/releases/download/v0.0/RVRT_supplementary.pdf)
**|** 
[pretrained models](https://github.com/JingyunLiang/RVRT/releases)
**|** 
[visual results](https://github.com/JingyunLiang/RVRT/releases)
**|** 
[original project page](https://github.com/JingyunLiang/RVRT)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2206.02146)
[![GitHub Stars](https://img.shields.io/github/stars/JingyunLiang/RVRT?style=social)](https://github.com/JingyunLiang/RVRT)
[![download](https://img.shields.io/github/downloads/JingyunLiang/RVRT/total.svg)](https://github.com/JingyunLiang/RVRT/releases)
![visitors](https://visitor-badge.glitch.me/badge?page_id=jingyunliang/RVRT)
[ <a href="https://colab.research.google.com/gist/JingyunLiang/23502e2c65d82144219fa3e3322e4fc3/rvrt-demo-on-video-restoration.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/gist/JingyunLiang/23502e2c65d82144219fa3e3322e4fc3/rvrt-demo-on-video-restoration.ipynb)

This is the readme of "Recurrent Video Restoration Transformer with Guided Deformable Attention"
([arxiv](https://arxiv.org/pdf/2206.02146.pdf), [supp](https://github.com/JingyunLiang/RVRT/releases/download/v0.0/RVRT_supplementary.pdf), [pretrained models](https://github.com/JingyunLiang/RVRT/releases), [visual results](https://github.com/JingyunLiang/RVRT/releases)). RVRT achieves state-of-the-art performance with balanced model size, testing memory and runtime in
- video SR (REDS, Vimeo90K, Vid4, UDM10) 
- video deblurring (GoPro, DVD)
- video denoising (DAVIS, Set8)

<p align="center">
  <a href="https://github.com/JingyunLiang/RVRT/releases">
    <img width=30% src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/teaser_vsr.gif"/>
    <img width=30% src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/teaser_vdb.gif"/>
    <img width=30% src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/teaser_vdn.gif"/>
  </a>
</p>

---

> Video restoration aims at restoring multiple high-quality frames from multiple low-quality frames. Existing video restoration methods generally fall into two extreme cases, i.e., they either restore all frames in parallel or restore the video frame by frame in a recurrent way, which would result in different merits and drawbacks. Typically, the former has the advantage of temporal information fusion. However, it suffers from large model size and intensive memory consumption; the latter has a relatively small model size as it shares parameters across frames; however, it lacks long-range dependency modeling ability and parallelizability. In this paper, we attempt to integrate the advantages of the two cases by proposing a recurrent video restoration transformer, namely RVRT. RVRT processes local neighboring frames in parallel within a globally recurrent framework which can achieve a good trade-off between model size, effectiveness, and efficiency. Specifically, RVRT divides the video into multiple clips and uses the previously inferred clip feature to estimate the subsequent clip feature. Within each clip, different frame features are jointly updated with implicit feature aggregation. Across different clips, the guided deformable attention is designed for clip-to-clip alignment, which predicts multiple relevant locations from the whole inferred clip and aggregates their features by the attention mechanism. Extensive experiments on video super-resolution, deblurring, and denoising show that the proposed RVRT achieves state-of-the-art performance on benchmark datasets with balanced model size, testing memory and runtime.
<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/rvrt.jpeg"> <br>
  <img height="200" src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/rfr.jpeg">
  <img height="200" src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/gda.jpeg">
</p>

#### Contents

1. [Requirements](#Requirements)
1. [Quick Testing](#Quick-Testing)
1. [Training](#Training)
1. [Results](#Results)
1. [Citation](#Citation)
1. [License and Acknowledgement](#License-and-Acknowledgement)


## Requirements
> - Python 3.8, PyTorch >= 1.9.1
> - Requirements: see requirements.txt
> - Platforms: Ubuntu 18.04, cuda-11.1

## Quick Testing
Following commands will download [pretrained models](https://github.com/JingyunLiang/RVRT/releases) and [test datasets](https://github.com/JingyunLiang/VRT/releases) **automatically** (except Vimeo-90K testing set). If out-of-memory, try to reduce `--tile` at the expense of slightly decreased performance. 

You can also try to test it on Colab[ <a href="https://colab.research.google.com/gist/JingyunLiang/23502e2c65d82144219fa3e3322e4fc3/rvrt-demo-on-video-restoration.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/gist/JingyunLiang/23502e2c65d82144219fa3e3322e4fc3/rvrt-demo-on-video-restoration.ipynb), but the results may be slightly different due to `--tile` difference.
```bash
# download code
git clone https://github.com/JingyunLiang/RVRT
cd RVRT
pip install -r requirements.txt

# 001, video sr trained on REDS, tested on REDS4
python main_test_rvrt.py --task 001_RVRT_videosr_bi_REDS_30frames --folder_lq testsets/REDS4/sharp_bicubic --folder_gt testsets/REDS4/GT --tile 100 128 128 --tile_overlap 2 20 20

# 002, video sr trained on Vimeo (bicubic), tested on Vid4 and Vimeo
python main_test_rvrt.py --task 002_RVRT_videosr_bi_Vimeo_14frames --folder_lq testsets/Vid4/BIx4 --folder_gt testsets/Vid4/GT --tile 0 0 0 --tile_overlap 2 20 20
python main_test_rvrt.py --task 002_RVRT_videosr_bi_Vimeo_14frames --folder_lq testsets/vimeo90k/vimeo_septuplet_matlabLRx4/sequences --folder_gt testsets/vimeo90k/vimeo_septuplet/sequences --tile 0 0 0 --tile_overlap 0 20 20

# 003, video sr trained on Vimeo (blur-downsampling), tested on Vid4, UDM10 and Vimeo
python main_test_rvrt.py --task 003_RVRT_videosr_bd_Vimeo_14frames --folder_lq testsets/Vid4/BDx4 --folder_gt testsets/Vid4/GT --tile 0 0 0 --tile_overlap 2 20 20
python main_test_rvrt.py --task 003_RVRT_videosr_bd_Vimeo_14frames --folder_lq testsets/UDM10/BDx4 --folder_gt testsets/UDM10/GT --tile 0 0 0 --tile_overlap 2 20 20
python main_test_rvrt.py --task 003_RVRT_videosr_bd_Vimeo_14frames --folder_lq testsets/vimeo90k/vimeo_septuplet_BDLRx4/sequences --folder_gt testsets/vimeo90k/vimeo_septuplet/sequences --tile 0 0 0 --tile_overlap 0 20 20

# 004, video deblurring trained and tested on DVD
python main_test_rvrt.py --task 004_RVRT_videodeblurring_DVD_16frames --folder_lq testsets/DVD10/test_GT_blurred --folder_gt testsets/DVD10/test_GT --tile 0 256 256 --tile_overlap 2 20 20

# 005, video deblurring trained and tested on GoPro
python main_test_rvrt.py --task 005_RVRT_videodeblurring_GoPro_16frames --folder_lq testsets/GoPro11/test_GT_blurred --folder_gt testsets/GoPro11/test_GT --tile 0 256 256 --tile_overlap 2 20 20

# 006, video denoising trained on DAVIS (noise level 0-50) and tested on Set8 and DAVIS
python main_test_rvrt.py --task 006_RVRT_videodenoising_DAVIS_16frames --sigma 50 --folder_lq testsets/Set8 --folder_gt testsets/Set8 --tile 0 256 256 --tile_overlap 2 20 20
python main_test_rvrt.py --task 006_RVRT_videodenoising_DAVIS_16frames --sigma 50  --folder_lq testsets/DAVIS-test --folder_gt testsets/DAVIS-test --tile 0 256 256 --tile_overlap 2 20 20

# test on your own datasets (an example)
python main_test_rvrt.py --task 001_RVRT_videosr_bi_REDS_30frames --folder_lq testsets/your/own --tile 0 0 0 --tile_overlap 2 20 20
```

**All visual results of RVRT can be downloaded [here](https://github.com/JingyunLiang/RVRT/releases)**.


## Training
The training and testing sets are as follows (see the [supplementary](https://github.com/JingyunLiang/RVRT/releases) for a detailed introduction of all datasets). For better I/O speed, use [create_lmdb.py](https://github.com/cszn/KAIR/tree/master/scripts/data_preparation/create_lmdb.py) to convert `.png` datasets to `.lmdb` datasets.

Note: You do **NOT need** to prepare the datasets if you just want to test the model. `main_test_rvrt.py` will download the testing set automaticaly.


| Task                                                          |                                                                                                                                                                                                                                    Training Set                                                                                                                                                                                                                                     |                                                                                                                                                                                                                                                                                 Testing Set                                                                                                                                                                                                                                                                                  |        Pretrained Model and Visual Results of RVRT  |
|:--------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|    :---:      |
| video SR (setting 1, BI)                                      |                                                                                 [REDS sharp & sharp_bicubic](https://seungjunnah.github.io/Datasets/reds.html) (266 videos, 266000 frames: train + val except REDS4)   <br  /><br  /> *Use  [regroup_reds_dataset.py](https://github.com/cszn/KAIR/tree/master/scripts/data_preparation/regroup_reds_dataset.py) to regroup and rename REDS val set                                                                                 |                                                                                                                                                                                                                                                           REDS4 (4 videos, 400 frames: 000, 011, 015, 020 of REDS)                                                                                                                                                                                                                                                           | [here](https://github.com/JingyunLiang/RVRT/releases) |
| video SR (setting 2 & 3, BI & BD)                             |    [Vimeo90K](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip) (64612 seven-frame videos as in `sep_trainlist.txt`)  <br  /><br  /> * Use [generate_LR_Vimeo90K.m](https://github.com/cszn/KAIR/tree/master/scripts/matlab_scripts/generate_LR_Vimeo90K.m) and [generate_LR_Vimeo90K_BD.m](https://github.com/cszn/KAIR/tree/master/scripts/matlab_scripts/generate_LR_Vimeo90K_BD.m) to generate LR frames for bicubic and blur-downsampling VSR, respectively.    |                                                                       Vimeo90K-T (the rest 7824 7-frame videos) + [Vid4](https://drive.google.com/file/d/1ZuvNNLgR85TV_whJoHM7uVb-XW1y70DW/view) (4 videos) + [UDM10](https://www.terabox.com/web/share/link?surl=LMuQCVntRegfZSxn7s3hXw&path=%2Fproject%2Fpfnl) (10 videos)  <br  /><br  /> *Use [prepare_UDM10.py](https://github.com/cszn/KAIR/tree/master/scripts/data_preparation/prepare_UDM10.py) to regroup and rename the UDM10 dataset                                                                        | [here](https://github.com/JingyunLiang/RVRT/releases) |
| video deblurring (setting 1, motion blur)                     |                                                                                            [DVD](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip) (61 videos, 5708 frames)  <br  /><br  /> *Use [prepare_DVD.py](https://github.com/cszn/KAIR/tree/master/scripts/data_preparation/prepare_DVD.py) to regroup and rename the dataset.                                                                                             |                                                                                                                                                                              DVD (10 videos, 1000 frames)             <br  /><br  /> *Use [evaluate_video_deblurring.m](https://github.com/cszn/KAIR/tree/master/scripts/matlab_scripts/evaluate_video_deblurring.m) for final evaluation.                                                                                                                                                                              | [here](https://github.com/JingyunLiang/RVRT/releases) |
| video deblurring (setting 2, motion blur)                     |                                                                                             [GoPro](http://data.cv.snu.ac.kr:8008/webdav/dataset/GOPRO/GOPRO_Large.zip) (22 videos, 2103 frames)  <br  /><br  /> *Use [prepare_GoPro_as_video.py](https://github.com/cszn/KAIR/tree/master/scripts/data_preparation/prepare_GoPro_as_video.py) to regroup and rename the dataset.                                                                                              |                                                                                                                                                                                  GoPro (11 videos, 1111 frames)  <br  /><br  /> *Use [evaluate_video_deblurring.m](https://github.com/cszn/KAIR/tree/master/scripts/matlab_scripts/evaluate_video_deblurring.m) for final evaluation.                                                                                                                                                                                   | [here](https://github.com/JingyunLiang/RVRT/releases) |
| video denoising (Gaussian noise)                              |                                                                                                                                             [DAVIS-2017](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip) (90 videos, 6208 frames)  <br  /><br  /> *Use all files in DAVIS/JPEGImages/480p                                                                                                                                              |                                                                                                              [DAVIS-2017-test](https://github.com/JingyunLiang/RVRT/releases) (30 videos) + [Set8](https://www.dropbox.com/sh/20n4cscqkqsfgoj/AABGftyJuJDwuCLGczL-fKvBa/test_sequences?dl=0&subfolder_nav_tracking=1) (8 videos: tractor, touchdown, park_joy and sunflower selected from DERF + hypersmooth, motorbike, rafting and snowboard from GOPRO_540P)                                                                                                               | [here](https://github.com/JingyunLiang/RVRT/releases) |

Run following commands for training:
```bash
# download code
git clone https://github.com/cszn/KAIR
cd KAIR
pip install -r requirements.txt

# 001, video sr trained on REDS, tested on REDS4
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_vrt.py --opt options/rvrt/001_train_rvrt_videosr_bi_reds_30frames.json  --dist True

# 002, video sr trained on Vimeo (bicubic), tested on Vid4 and Vimeo
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_vrt.py --opt options/rvrt/002_train_rvrt_videosr_bi_vimeo_14frames.json  --dist True

# 003, video sr trained on Vimeo (blur-downsampling), tested on Vid4, Vimeo and UDM10
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_vrt.py --opt options/rvrt/003_train_rvrt_videosr_bd_vimeo_14frames.json  --dist True

# 004, video deblurring trained and tested on DVD
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_vrt.py --opt options/rvrt/004_train_rvrt_videodeblurring_dvd.json  --dist True

# 005, video deblurring trained and tested on GoPro
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_vrt.py --opt options/rvrt/005_train_rvrt_videodeblurring_gopro.json  --dist True

# 006, video denoising trained on DAVIS (noise level 0-50) and tested on Set8 and DAVIS
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_vrt.py --opt options/rvrt/006_train_rvrt_videodenoising_davis.json  --dist True
```
Tip: The training process will terminate automatically at 20000 iteration due to a bug. Just resume training after that.
<details>
<summary>Bug</summary>
Bug: PyTorch DistributedDataParallel (DDP) does not support `torch.utils.checkpoint` well. To alleviate the problem, set `find_unused_parameters=False` when `use_checkpoint=True`. If there are other errors, make sure that unused parameters will not change during training loop and set `use_static_graph=True`.

If you find a better solution, feel free to pull a request. Thank you.
</details>

## Results
We achieved state-of-the-art performance on video SR, video deblurring and video denoising. Detailed results can be found in the [paper](https://arxiv.org/abs/2206.02146).

<details>
<summary>Video Super-Resolution (click me)</summary>
<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/vsr.jpg">
  <img width="800" src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/vsr_visual.jpg">
</p>
</details>

<details>
<summary>Video Deblurring</summary>
<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/vdb.jpg">
  <img width="800" src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/vdb_visual.jpg">
</p>
</details>

<details>
<summary>Video Denoising</summary>
<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/vdn.jpg">
  <img width="800" src="https://raw.githubusercontent.com/JingyunLiang/RVRT/main/assets/vdn_visual.jpg">
</p>
</details>


## Citation
    @article{liang2022rvrt,
        title={Recurrent Video Restoration Transformer with Guided Deformable Attention},
        author={Liang, Jingyun and Fan, Yuchen and Xiang, Xiaoyu and Ranjan, Rakesh and Ilg, Eddy  and Green, Simon and Cao, Jiezhang and Zhang, Kai and Timofte, Radu and Van Gool, Luc},
        journal={arXiv preprint arXiv:2206.02146},
        year={2022}
    }


## License and Acknowledgement
This project is released under the CC-BY-NC license. We refer to codes from [KAIR](https://github.com/cszn/KAIR), [BasicSR](https://github.com/xinntao/BasicSR), [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) and [mmediting](https://github.com/open-mmlab/mmediting). Thanks for their awesome works. The majority of VRT is licensed under CC-BY-NC, however portions of the project are available under separate license terms: KAIR is licensed under the MIT License, BasicSR, Video Swin Transformer and mmediting are licensed under the Apache 2.0 license.