import os.path
import logging
import time
from collections import OrderedDict
import torch

from utils import utils_logger
from utils import utils_image as util
# from utils import utils_model


'''
This code can help you to calculate:
`FLOPs`, `#Params`, `Runtime`, `#Activations`, `#Conv`, and `Max Memory Allocated`.

- `#Params' denotes the total number of parameters. 
- `FLOPs' is the abbreviation for floating point operations. 
- `#Activations' measures the number of elements of all outputs of convolutional layers. 
- `Memory' represents maximum GPU memory consumption according to the PyTorch function torch.cuda.max_memory_allocated().
- `#Conv' represents the number of convolutional layers. 
- `FLOPs', `#Activations', and `Memory' are tested on an LR image of size 256x256.

For more information, please refer to ECCVW paper "AIM 2020 Challenge on Efficient Super-Resolution: Methods and Results".

# If you use this code, please consider the following citations:

@inproceedings{zhang2020aim,
  title={AIM 2020 Challenge on Efficient Super-Resolution: Methods and Results},
  author={Kai Zhang and Martin Danelljan and Yawei Li and Radu Timofte and others},
  booktitle={European Conference on Computer Vision Workshops},
  year={2020}
}
@inproceedings{zhang2019aim,
  title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
  author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  year={2019}
}

CuDNN (https://developer.nvidia.com/rdp/cudnn-archive) should be installed.

For `Memery` and `Runtime`, set 'print_modelsummary = False' and 'save_results = False'.
'''




def main():

    utils_logger.logger_info('efficientsr_challenge', log_path='efficientsr_challenge.log')
    logger = logging.getLogger('efficientsr_challenge')

#    print(torch.__version__)               # pytorch version
#    print(torch.version.cuda)              # cuda version
#    print(torch.backends.cudnn.version())  # cudnn version

    # --------------------------------
    # basic settings
    # --------------------------------
    model_names = ['msrresnet', 'imdn']
    model_id = 1                  # set the model name
    sf = 4
    model_name = model_names[model_id]
    logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

    testsets = 'testsets'         # set path of testsets
    testset_L = 'DIV2K_valid_LR'  # set current testing dataset; 'DIV2K_test_LR'
    testset_L = 'set12'

    save_results = True
    print_modelsummary = True     # set False when calculating `Max Memery` and `Runtime`

    torch.cuda.set_device(0)      # set GPU ID
    logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # define network and load model
    # --------------------------------
    if model_name == 'msrresnet':
        from models.network_msrresnet import MSRResNet1 as net
        model = net(in_nc=3, out_nc=3, nc=64, nb=16, upscale=4)  # define network
        model_path = os.path.join('model_zoo', 'msrresnet_x4_psnr.pth')  # set model path
    elif model_name == 'imdn':
        from models.network_imdn import IMDN as net
        model = net(in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle')  # define network
        model_path = os.path.join('model_zoo', 'imdn_x4.pth')            # set model path

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # --------------------------------
    # print model summary
    # --------------------------------
    if print_modelsummary:
        from utils.utils_modelsummary import get_model_activation, get_model_flops
        input_dim = (3, 256, 256)  # set the input dimension

        activations, num_conv2d = get_model_activation(model, input_dim)
        logger.info('{:>16s} : {:<.4f} [M]'.format('#Activations', activations/10**6))
        logger.info('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))

        flops = get_model_flops(model, input_dim, False)
        logger.info('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops/10**9))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        logger.info('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))

    # --------------------------------
    # read image
    # --------------------------------
    L_path = os.path.join(testsets, testset_L)
    E_path = os.path.join(testsets, testset_L+'_'+model_name)
    util.mkdir(E_path)

    # record runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
    logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for img in util.get_image_paths(L_path):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        torch.cuda.empty_cache()
        img_L = img_L.to(device)

        start.record()
        img_E = model(img_L)
        # img_E = utils_model.test_mode(model, img_L, mode=2, min_size=480, sf=sf)  # use this to avoid 'out of memory' issue.
        # logger.info('{:>16s} : {:<.3f} [M]'.format('Max Memery', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))  # Memery
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds


#        torch.cuda.synchronize()
#        start = time.time()
#        img_E = model(img_L)
#        torch.cuda.synchronize()
#        end = time.time()
#        test_results['runtime'].append(end-start)  # seconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)

        if save_results:
            util.imsave(img_E, os.path.join(E_path, img_name+ext))
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_path, ave_runtime))


if __name__ == '__main__':

    main()
