'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
https://github.com/yangxy/GPEN
@inproceedings{Yang2021GPEN,
    title={GAN Prior Embedded Network for Blind Face Restoration in the Wild},
    author={Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}
© Alibaba, 2021. For academic and non-commercial use only.
==================================================
slightly modified by Kai Zhang (2021-06-03)
https://github.com/cszn/KAIR

How to run:

step 1: Download <RetinaFace-R50.pth> model and <GPEN-512.pth> model and put them into `model_zoo`.
RetinaFace-R50.pth: https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth
GPEN-512.pth: https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-512.pth

step 2: Install ninja by `pip install ninja`; set <inputdir> for your own testing images

step 3: `python main_test_face_enhancement.py`
==================================================
'''


import os
import cv2
import glob
import numpy as np
import torch

from utils.utils_alignfaces import warp_and_crop_face, get_reference_facial_points
from utils import utils_image as util 

from retinaface.retinaface_detection import RetinaFaceDetection
from models.network_faceenhancer import FullGenerator as enhancer_net


class faceenhancer(object):
    def __init__(self, model_path='model_zoo/GPEN-512.pth', size=512, channel_multiplier=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.size = size
        self.model = enhancer_net(self.size, 512, 8, channel_multiplier).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def process(self, img):
        '''
        img: uint8 RGB image, (W, H, 3)
        out: uint8 RGB image, (W, H, 3)
        '''
        img = cv2.resize(img, (self.size, self.size))
        img = util.uint2tensor4(img)
        img = (img - 0.5) / 0.5
        img = img.to(self.device)

        with torch.no_grad():
            out, __ = self.model(img)

        out = util.tensor2uint(out * 0.5 + 0.5)
        return out


class faceenhancer_with_detection_alignment(object):
    def __init__(self, model_path, size=512, channel_multiplier=2):
        self.facedetector = RetinaFaceDetection('model_zoo/RetinaFace-R50.pth')
        self.faceenhancer = faceenhancer(model_path, size, channel_multiplier)
        self.size = size
        self.threshold = 0.9

        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)

        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
                (self.size, self.size), inner_padding_factor, outer_padding, default_square)

    def process(self, img):
        '''
        img: uint8 RGB image, (W, H, 3)
        img, orig_faces, enhanced_faces: uint8 RGB image / cropped face images
        '''
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        facebs, landms = self.facedetector.detect(img_bgr)

        orig_faces, enhanced_faces = [], []
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4]<self.threshold: continue
            fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.size, self.size))
            # Enhance the face image!
            
            ef = self.faceenhancer.process(of)

            orig_faces.append(of)
            enhanced_faces.append(ef)
            tmp_mask = self.mask
            tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)
            if min(fh, fw) < 100: # Gaussian filter for small face
                ef = cv2.filter2D(ef, -1, self.kernel)

            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
            full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]

        full_mask = full_mask[:, :, np.newaxis]
        img = cv2.convertScaleAbs(img*(1-full_mask) + full_img*full_mask)

        return img, orig_faces, enhanced_faces


if __name__=='__main__':

    inputdir = os.path.join('testsets', 'real_faces')
    outdir = os.path.join('testsets', 'real_faces_results')
    os.makedirs(outdir, exist_ok=True)

    # whether use the face detection&alignment or not
    need_face_detection = True

    if need_face_detection:
        enhancer = faceenhancer_with_detection_alignment(model_path=os.path.join('model_zoo','GPEN-512.pth'), size=512, channel_multiplier=2)
    else:
        enhancer = faceenhancer(model_path=os.path.join('model_zoo','GPEN-512.pth'), size=512, channel_multiplier=2)

    for idx, img_file in enumerate(util.get_image_paths(inputdir)):
        img_name, ext = os.path.splitext(os.path.basename(img_file))
        img_L = util.imread_uint(img_file, n_channels=3)

        print('{:->4d} --> {:<s}'.format(idx+1, img_name+ext))

        img_L = cv2.resize(img_L, (0,0), fx=2, fy=2)

        if need_face_detection:
            # do the enhancement
            img_H, orig_faces, enhanced_faces = enhancer.process(img_L)

            util.imsave(np.hstack((img_L, img_H)), os.path.join(outdir, img_name+'_comparison.png'))
            util.imsave(img_H, os.path.join(outdir, img_name+'_enhanced.png'))
            for m, (ef, of) in enumerate(zip(enhanced_faces, orig_faces)):
                of = cv2.resize(of, ef.shape[:2])
                util.imsave(np.hstack((of, ef)), os.path.join(outdir, img_name+'_face%02d'%m+'.png'))
        else:
            # do the enhancement
            img_H = enhancer.process(img_L)

            util.imsave(img_H, os.path.join(outdir, img_name+'_enhanced_without_detection.png'))
