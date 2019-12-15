import torch.utils.data as data
import utils.utils_image as util


class DatasetL(data.Dataset):
    '''
    # -----------------------------------------
    # Get L in testing.
    # Only "dataroot_L" is needed.
    # -----------------------------------------
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetL, self).__init__()
        print('Read L in testing. Only "dataroot_L" is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3

        # ------------------------------------
        # get the path of L
        # ------------------------------------
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        assert self.paths_L, 'Error: L paths are empty.'

    def __getitem__(self, index):
        L_path = None

        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)

        # ------------------------------------
        # HWC to CHW, numpy to tensor
        # ------------------------------------
        img_L = util.uint2tensor3(img_L)

        return {'L': img_L, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_L)
