import os
import cv2
import glob
import numpy as np
import torch

from models.network_faceenhancer import FullGenerator as enhancer_net
from utils import utils_image as util


def finetune(model, epochs, imgs, target_imgs, optimizer, criterion, device, size):
    model.train()
    for epoch in range(1, epochs):
        for batch_idx, (img, target_img) in enumerate(zip(imgs, target_imgs)):
            img = cv2.resize(img, (size, size))
            img = util.uint2tensor4(img)
            img = (img - 0.5) / 0.5
            img = img.to(device)
            optimizer.zero_grad()

            output = model(img)
            loss = criterion(output, target_img)
            loss.backward()

            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')



def main():
    model_path = 'model_zoo/GPEN-512.pth'
    finetuned_model_path = 'model_zoo/GPEN-512-finetuned.pth'
    size = 512
    channel_multiplier = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 1

    model = enhancer_net(size, 512, 8, channel_multiplier).to(device)
    model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    input_imgs = []
    inputdir = os.path.join('testsets', 'dirk_original_recreated_deepfake')
    for idx, img_file in enumerate(util.get_image_paths(inputdir)):
        img_name, ext = os.path.splitext(os.path.basename(img_file))
        img_L = util.imread_uint(img_file, n_channels=3)
        print('{:->4d} --> {:<s}'.format(idx+1, img_name+ext))
        img_L = cv2.resize(img_L, (0,0), fx=2, fy=2)
        input_imgs.append(img_L)

    target_imgs = []
    targetdir = os.path.join('testsets', 'dirk_original')
    for idx, img_file in enumerate(util.get_image_paths(targetdir)):
        img_name, ext = os.path.splitext(os.path.basename(img_file))
        img_L = util.imread_uint(img_file, n_channels=3)
        print('{:->4d} --> {:<s}'.format(idx + 1, img_name + ext))
        img_L = cv2.resize(img_L, (0, 0), fx=2, fy=2)
        target_imgs.append(img_L)

    finetune(model, epochs, input_imgs, target_imgs, optimizer, criterion, device, size)

    torch.save(model.state_dict(), finetuned_model_path)



if __name__ == '__main__':
    main()
