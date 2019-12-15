## Training and testing codes for DnCNN, FFDNet, SRMD, DPSR, MSRResNet, ESRGAN, IMDN
### Training
- [main_train_dncnn.py](main_train_dncnn.py)
- [main_train_fdncnn.py](main_train_fdncnn.py)
- [main_train_ffdnet.py](main_train_ffdnet.py)
- [main_train_srmd.py](main_train_srmd.py)
- [main_train_dpsr.py](main_train_dpsr.py)
- [main_train_msrresnet_psnr.py](main_train_msrresnet_psnr.py)
- [main_train_msrresnet_gan.py](main_train_msrresnet_gan.py)
- [main_train_rrdb_psnr.py](main_train_rrdb_psnr.py)
- [main_train_imdn.py](main_train_imdn.py)
### Testing
- [main_test_dncnn.py](main_test_dncnn.py)
- [main_test_fdncnn.py](main_test_fdncnn.py)
- [main_test_ffdnet.py](main_test_ffdnet.py)
- [main_test_srmd.py](main_test_srmd.py)
- [main_test_dpsr.py](main_test_dpsr.py)
- [main_test_msrresnet.py](main_test_msrresnet.py)
- [main_test_rrdb.py](main_test_rrdb.py)
- [main_test_imdn.py](main_test_imdn.py)

### References
```
@article{zhang2017beyond, % DnCNN
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017}
}
@article{zhang2018ffdnet, % FFDNet, FDnCNN
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018}
}
@inproceedings{zhang2018learning, % SRMD
  title={Learning a single convolutional super-resolution network for multiple degradations},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3262--3271},
  year={2018}
}
@inproceedings{zhang2019deep, % DPSR
  title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1671--1681},
  year={2019}
}
@InProceedings{wang2018esrgan, % ESRGAN, MSRResNet
    author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
    title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
    booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
    month = {September},
    year = {2018}
}
@inproceedings{hui2019lightweight, % IMDN
  title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
  author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
  pages={2024--2032},
  year={2019}
}
@inproceedings{zhang2019aim, % IMDN
  title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
  author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  year={2019}
}
```
