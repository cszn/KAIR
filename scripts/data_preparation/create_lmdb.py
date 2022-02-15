import argparse
from os import path as osp

from utils.utils_video import scandir
from utils.utils_lmdb import make_lmdb_from_imgs


def create_lmdb_for_div2k():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR images
    folder_path = 'trainsets/DIV2K/DIV2K_train_HR_sub'
    lmdb_path = 'trainsets/DIV2K/DIV2K_train_HR_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx2 images
    folder_path = 'trainsets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
    lmdb_path = 'trainsets/DIV2K/DIV2K_train_LR_bicubic_X2_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx3 images
    folder_path = 'trainsets/DIV2K/DIV2K_train_LR_bicubic/X3_sub'
    lmdb_path = 'trainsets/DIV2K/DIV2K_train_LR_bicubic_X3_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx4 images
    folder_path = 'trainsets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    lmdb_path = 'trainsets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def create_lmdb_for_reds():
    """Create lmdb files for REDS dataset.

    Usage:
        Before run this script, please run `regroup_reds_dataset.py`.
        We take three folders for example:
            train_sharp
            train_sharp_bicubic
            train_blur (for video deblurring)
        Remember to modify opt configurations according to your settings.
    """
    # train_sharp
    folder_path = 'trainsets/REDS/train_sharp'
    lmdb_path = 'trainsets/REDS/train_sharp_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # train_sharp_bicubic
    folder_path = 'trainsets/REDS/train_sharp_bicubic'
    lmdb_path = 'trainsets/REDS/train_sharp_bicubic_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # train_blur (for video deblurring)
    folder_path = 'trainsets/REDS_blur/train_blur'
    lmdb_path = 'trainsets/REDS_blur/train_blur_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # train_blur_bicubic (for video deblurring-sr)
    folder_path = 'trainsets/REDS_blur_bicubic/train_blur_bicubic'
    lmdb_path = 'trainsets/REDS_blur_bicubic/train_blur_bicubic_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_reds(folder_path):
    """Prepare image path list and keys for REDS dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=True)))
    keys = [v.split('.png')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys


def create_lmdb_for_vimeo90k():
    """Create lmdb files for Vimeo90K dataset.

    Usage:
        Remember to modify opt configurations according to your settings.
    """
    # GT
    folder_path = 'trainsets/vimeo90k/vimeo_septuplet/sequences'
    lmdb_path = 'trainsets/vimeo90k/vimeo90k_train_GT_only4th.lmdb'
    train_list_path = 'trainsets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path, 'gt')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # LQ
    folder_path = 'trainsets/vimeo90k/vimeo_septuplet_matlabLRx4/sequences'
    lmdb_path = 'trainsets/vimeo90k/vimeo90k_train_LR7frames.lmdb'
    train_list_path = 'trainsets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path, 'lq')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def create_lmdb_for_vimeo90k_bd():
    """Create lmdb files for Vimeo90K dataset (blur-downsampled lr only).

    Usage:
        Remember to modify opt configurations according to your settings.
    """
    # LQ (blur-downsampled, BD)
    folder_path = 'trainsets/vimeo90k/vimeo_septuplet_BDLRx4/sequences'
    lmdb_path = 'trainsets/vimeo90k/vimeo90k_train_BDLR7frames.lmdb'
    train_list_path = 'trainsets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path, 'lq')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_vimeo90k(folder_path, train_list_path, mode):
    """Prepare image path list and keys for Vimeo90K dataset.

    Args:
        folder_path (str): Folder path.
        train_list_path (str): Path to the official train list.
        mode (str): One of 'gt' or 'lq'.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    with open(train_list_path, 'r') as fin:
        train_list = [line.strip() for line in fin]

    img_path_list = []
    keys = []
    for line in train_list:
        folder, sub_folder = line.split('/')
        img_path_list.extend([osp.join(folder, sub_folder, f'im{j + 1}.png') for j in range(7)])
        keys.extend([f'{folder}/{sub_folder}/im{j + 1}' for j in range(7)])

    if mode == 'gt':
        print('Only keep the 4th frame for the gt mode.')
        img_path_list = [v for v in img_path_list if v.endswith('im4.png')]
        keys = [v for v in keys if v.endswith('/im4')]

    return img_path_list, keys


def create_lmdb_for_dvd():
    """Create lmdb files for DVD dataset.

    Usage:
        We take two folders for example:
            GT
            input
        Remember to modify opt configurations according to your settings.
    """
    # train_sharp
    folder_path = 'trainsets/DVD/train_GT'
    lmdb_path = 'trainsets/DVD/train_GT.lmdb'
    img_path_list, keys = prepare_keys_dvd(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # train_sharp_bicubic
    folder_path = 'trainsets/DVD/train_GT_blurred'
    lmdb_path = 'trainsets/DVD/train_GT_blurred.lmdb'
    img_path_list, keys = prepare_keys_dvd(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_dvd(folder_path):
    """Prepare image path list and keys for DVD dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='jpg', recursive=True)))
    keys = [v.split('.jpg')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys


def create_lmdb_for_gopro():
    """Create lmdb files for GoPro dataset.

    Usage:
        We take two folders for example:
            GT
            input
        Remember to modify opt configurations according to your settings.
    """
    # train_sharp
    folder_path = 'trainsets/GoPro/train_GT'
    lmdb_path = 'trainsets/GoPro/train_GT.lmdb'
    img_path_list, keys = prepare_keys_gopro(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # train_sharp_bicubic
    folder_path = 'trainsets/GoPro/train_GT_blurred'
    lmdb_path = 'trainsets/GoPro/train_GT_blurred.lmdb'
    img_path_list, keys = prepare_keys_gopro(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_gopro(folder_path):
    """Prepare image path list and keys for GoPro dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=True)))
    keys = [v.split('.png')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys


def create_lmdb_for_davis():
    """Create lmdb files for DAVIS dataset.

    Usage:
        We take one folders for example:
            GT
        Remember to modify opt configurations according to your settings.
    """
    # train_sharp
    folder_path = 'trainsets/DAVIS/train_GT'
    lmdb_path = 'trainsets/DAVIS/train_GT.lmdb'
    img_path_list, keys = prepare_keys_davis(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_davis(folder_path):
    """Prepare image path list and keys for DAVIS dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='jpg', recursive=True)))
    keys = [v.split('.jpg')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys



def create_lmdb_for_ldv():
    """Create lmdb files for LDV dataset.

    Usage:
        We take two folders for example:
            GT
            input
        Remember to modify opt configurations according to your settings.
    """
    # training_raw
    folder_path = 'trainsets/LDV/training_raw'
    lmdb_path = 'trainsets/LDV/training_raw.lmdb'
    img_path_list, keys = prepare_keys_ldv(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # training_fixed-QP
    folder_path = 'trainsets/LDV/training_fixed-QP'
    lmdb_path = 'trainsets/LDV/training_fixed-QP.lmdb'
    img_path_list, keys = prepare_keys_ldv(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # training_fixed-rate
    folder_path = 'trainsets/LDV/training_fixed-rate'
    lmdb_path = 'trainsets/LDV/training_fixed-rate.lmdb'
    img_path_list, keys = prepare_keys_ldv(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_ldv(folder_path):
    """Prepare image path list and keys for LDV dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=True)))
    keys = [v.split('.png')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys


def create_lmdb_for_reds_orig():
    """Create lmdb files for REDS_orig dataset (120 fps).

    Usage:
        Before run this script, please run `regroup_reds_dataset.py`.
        We take one folders for example:
            train_orig
        Remember to modify opt configurations according to your settings.
    """
    # train_sharp
    folder_path = 'trainsets/REDS_orig/train_orig'
    lmdb_path = 'trainsets/REDS_orig/train_orig_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds_orig(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_reds_orig(folder_path):
    """Prepare image path list and keys for REDS_orig dataset (120 fps).

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=True)))
    keys = [v.split('.png')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        help=("Options: 'DIV2K', 'REDS', 'Vimeo90K', 'Vimeo90K_BD', 'DVD', 'GoPro',"
              "'DAVIS', 'LDV', 'REDS_orig' "
              'You may need to modify the corresponding configurations in codes.'))
    args = parser.parse_args()
    dataset = args.dataset.lower()
    if dataset == 'div2k':
        create_lmdb_for_div2k()
    elif dataset == 'reds':
        create_lmdb_for_reds()
    elif dataset == 'vimeo90k':
        create_lmdb_for_vimeo90k()
    elif dataset == 'vimeo90k_bd':
        create_lmdb_for_vimeo90k_bd()
    elif dataset == 'dvd':
        create_lmdb_for_dvd()
    elif dataset == 'gopro':
        create_lmdb_for_gopro()
    elif dataset == 'davis':
        create_lmdb_for_davis()
    elif dataset == 'ldv':
        create_lmdb_for_ldv()
    elif dataset == 'reds_orig':
        create_lmdb_for_reds_orig()
    else:
        raise ValueError('Wrong dataset.')
