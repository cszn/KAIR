import os
import glob
import shutil


def generate_meta_info_txt(data_path, meta_info_path):
    '''generate meta_info_DAVIS_GT.txt for DAVIS

    :param data_path: dataset path.
    :return: None
    '''
    f= open(meta_info_path, "w+")
    file_list = sorted(glob.glob(os.path.join(data_path, 'train_GT/*')))
    total_frames = 0
    for path in file_list:
        name = os.path.basename(path)
        frames = sorted(glob.glob(os.path.join(path, '*')))
        start_frame = os.path.basename(frames[0]).split('.')[0]

        print(name, len(frames), start_frame)
        total_frames += len(frames)

        f.write(f"{name} {len(frames)} (480,854,3) {start_frame}\r\n")

    assert total_frames == 6208, f'DAVIS training set should have 6208 images, but got {total_frames} images'

if __name__ == '__main__':

    dataset_path = 'trainsets/DAVIS'

    generate_meta_info_txt(dataset_path, 'data/meta_info/meta_info_DAVIS_GT.txt')


