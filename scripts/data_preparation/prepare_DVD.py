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
    os.makedirs(os.path.join(dataset_path, 'train_GT_with_val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'train_GT_blurred_with_val'), exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(dataset_path, '*')))
    for path in file_list:
        if 'train_GT_with_val' in path or 'train_GT_blurred_with_val' in path:
            continue
        name = os.path.basename(path)
        print(name)

        shutil.move(os.path.join(path, 'GT'), os.path.join(f'{dataset_path}/train_GT_with_val', name))
        shutil.move(os.path.join(path, 'input'), os.path.join(f'{dataset_path}/train_GT_blurred_with_val', name))
        shutil.rmtree(path)


def generate_meta_info_txt(data_path, meta_info_path):
    '''generate meta_info_DVD_GT.txt for DVD

    :param data_path: dataset path.
    :return: None
    '''
    f= open(meta_info_path, "w+")
    file_list = sorted(glob.glob(os.path.join(data_path, 'train_GT_with_val/*')))
    total_frames = 0
    for path in file_list:
        name = os.path.basename(path)
        frames = sorted(glob.glob(os.path.join(path, '*')))
        start_frame = os.path.basename(frames[0]).split('.')[0]

        print(name, len(frames), start_frame)
        total_frames += len(frames)

        f.write(f"{name} {len(frames)} (720,1280,3) {start_frame}\r\n")

    assert total_frames == 6708, f'DVD training+Validation set should have 6708 images, but got {total_frames} images'


if __name__ == '__main__':

    dataset_path = 'trainsets/DeepVideoDeblurring_Dataset/quantitative_datasets'

    rearrange_dir_structure(dataset_path)
    generate_meta_info_txt(dataset_path, 'data/meta_info/meta_info_DVD_GT.txt')


