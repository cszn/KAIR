import os
import glob
import shutil


def rearrange_dir_structure(dataset_path):
    '''move files to follow the directory structure as REDS

    Original DVD dataset is organized as DVD/quantitative_datasets/720p_240fps_1/GT/00000.jpg.
    We move files and organize them as DVD/train_GT_with_val/720p_240fps_1/00000.jpg (similar to REDS).

    :param dataset_path: dataset path
    :return: None
    '''
    os.makedirs(os.path.join(dataset_path, 'GT'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'BDx4'), exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(dataset_path, '*')))
    for path in file_list:
        if 'GT' in path or 'BDx4' in path:
            continue
        name = os.path.basename(path)
        print(name)

        shutil.move(os.path.join(path, 'truth'), os.path.join(f'{dataset_path}/GT', name))
        shutil.move(os.path.join(path, 'blur4'), os.path.join(f'{dataset_path}/BDx4', name))
        shutil.rmtree(path)


if __name__ == '__main__':

    dataset_path = 'trainsets/UDM10'

    rearrange_dir_structure(dataset_path)


