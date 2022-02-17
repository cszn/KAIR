import os
import cv2
import numpy as np
import torch
import random
from os import path as osp
from torchvision.utils import make_grid
import sys
from pathlib import Path
import six
from collections import OrderedDict
import math
import glob
import av
import io
from cv2 import (CAP_PROP_FOURCC, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT,
                 CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
                 CAP_PROP_POS_FRAMES, VideoWriter_fourcc)

if sys.version_info <= (3, 3):
    FileNotFoundError = IOError
else:
    FileNotFoundError = FileNotFoundError


def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)


def is_filepath(x):
    return is_str(x) or isinstance(x, Path)


def fopen(filepath, *args, **kwargs):
    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)
    raise ValueError('`filepath` should be a string or a Path')


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        case_sensitive (bool, optional) : If set to False, ignore the case of
            suffix. Default: True.
    Returns:
        A generator for all the interested files with relative paths.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = suffix.lower() if isinstance(suffix, str) else tuple(
            item.lower() for item in suffix)

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive,
                                    case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)


class Cache:

    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be a positive integer')

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key, val):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        val = self._cache[key] if key in self._cache else default
        return val


class VideoReader:
    """Video class with similar usage to a list object.

    This video warpper class provides convenient apis to access frames.
    There exists an issue of OpenCV's VideoCapture class that jumping to a
    certain frame may be inaccurate. It is fixed in this class by checking
    the position after jumping each time.
    Cache is used when decoding videos. So if the same frame is visited for
    the second time, there is no need to decode again if it is stored in the
    cache.

    """

    def __init__(self, filename, cache_capacity=10):
        # Check whether the video path is a url
        if not filename.startswith(('https://', 'http://')):
            check_file_exist(filename, 'Video file not found: ' + filename)
        self._vcap = cv2.VideoCapture(filename)
        assert cache_capacity > 0
        self._cache = Cache(cache_capacity)
        self._position = 0
        # get basic info
        self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
        self._fps = self._vcap.get(CAP_PROP_FPS)
        self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(CAP_PROP_FOURCC)

    @property
    def vcap(self):
        """:obj:`cv2.VideoCapture`: The raw VideoCapture object."""
        return self._vcap

    @property
    def opened(self):
        """bool: Indicate whether the video is opened."""
        return self._vcap.isOpened()

    @property
    def width(self):
        """int: Width of video frames."""
        return self._width

    @property
    def height(self):
        """int: Height of video frames."""
        return self._height

    @property
    def resolution(self):
        """tuple: Video resolution (width, height)."""
        return (self._width, self._height)

    @property
    def fps(self):
        """float: FPS of the video."""
        return self._fps

    @property
    def frame_cnt(self):
        """int: Total frames of the video."""
        return self._frame_cnt

    @property
    def fourcc(self):
        """str: "Four character code" of the video."""
        return self._fourcc

    @property
    def position(self):
        """int: Current cursor position, indicating frame decoded."""
        return self._position

    def _get_real_position(self):
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self):
        """Read the next frame.

        If the next frame have been decoded before and in the cache, then
        return it directly, otherwise decode, cache and return it.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        # pos = self._position
        if self._cache:
            img = self._cache.get(self._position)
            if img is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    self._cache.put(self._position, img)
        else:
            ret, img = self._vcap.read()
        if ret:
            self._position += 1
        return img

    def get_frame(self, frame_id):
        """Get frame by index.

        Args:
            frame_id (int): Index of the expected frame, 0-based.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(
                f'"frame_id" must be between 0 and {self._frame_cnt - 1}')
        if frame_id == self._position:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id + 1
                return img
        self._set_real_position(frame_id)
        ret, img = self._vcap.read()
        if ret:
            if self._cache:
                self._cache.put(self._position, img)
            self._position += 1
        return img

    def current_frame(self):
        """Get the current frame (frame that is just visited).

        Returns:
            ndarray or None: If the video is fresh, return None, otherwise
            return the frame.
        """
        if self._position == 0:
            return None
        return self._cache.get(self._position - 1)

    def cvt2frames(self,
                   frame_dir,
                   file_start=0,
                   filename_tmpl='{:06d}.jpg',
                   start=0,
                   max_num=0,
                   show_progress=False):
        """Convert a video to frame images.

        Args:
            frame_dir (str): Output directory to store all the frame images.
            file_start (int): Filenames will start from the specified number.
            filename_tmpl (str): Filename template with the index as the
                placeholder.
            start (int): The starting frame index.
            max_num (int): Maximum number of frames to be written.
            show_progress (bool): Whether to show a progress bar.
        """
        mkdir_or_exist(frame_dir)
        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError('start must be less than total frame number')
        if start > 0:
            self._set_real_position(start)

        def write_frame(file_idx):
            img = self.read()
            if img is None:
                return
            filename = osp.join(frame_dir, filename_tmpl.format(file_idx))
            cv2.imwrite(filename, img)

        if show_progress:
            pass
            #track_progress(write_frame, range(file_start,file_start + task_num))
        else:
            for i in range(task_num):
                write_frame(file_start + i)

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self.get_frame(i)
                for i in range(*index.indices(self.frame_cnt))
            ]
        # support negative indexing
        if index < 0:
            index += self.frame_cnt
            if index < 0:
                raise IndexError('index out of range')
        return self.get_frame(index)

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self):
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()


def frames2video(frame_dir,
                 video_file,
                 fps=30,
                 fourcc='XVID',
                 filename_tmpl='{:06d}.jpg',
                 start=0,
                 end=0,
                 show_progress=False):
    """Read the frame images from a directory and join them as a video.

    Args:
        frame_dir (str): The directory containing video frames.
        video_file (str): Output filename.
        fps (float): FPS of the output video.
        fourcc (str): Fourcc of the output video, this should be compatible
            with the output file type.
        filename_tmpl (str): Filename template with the index as the variable.
        start (int): Starting frame index.
        end (int): Ending frame index.
        show_progress (bool): Whether to show a progress bar.
    """
    if end == 0:
        ext = filename_tmpl.split('.')[-1]
        end = len([name for name in scandir(frame_dir, ext)])
    first_file = osp.join(frame_dir, filename_tmpl.format(start))
    check_file_exist(first_file, 'The start frame not found: ' + first_file)
    img = cv2.imread(first_file)
    height, width = img.shape[:2]
    resolution = (width, height)
    vwriter = cv2.VideoWriter(video_file, VideoWriter_fourcc(*fourcc), fps,
                              resolution)

    def write_frame(file_idx):
        filename = osp.join(frame_dir, filename_tmpl.format(file_idx))
        img = cv2.imread(filename)
        vwriter.write(img)

    if show_progress:
        pass
        # track_progress(write_frame, range(start, end))
    else:
        for i in range(start, end):
            write_frame(i)
    vwriter.release()


def video2images(video_path, output_dir):
    vidcap = cv2.VideoCapture(video_path)
    in_fps = vidcap.get(cv2.CAP_PROP_FPS)
    print('video fps:', in_fps)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    loaded, frame = vidcap.read()
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'number of total frames is: {total_frames:06}')
    for i_frame in range(total_frames):
        if i_frame % 100 == 0:
            print(f'{i_frame:06} / {total_frames:06}')
        frame_name = os.path.join(output_dir, f'{i_frame:06}' + '.png')
        cv2.imwrite(frame_name, frame)
        loaded, frame = vidcap.read()


def images2video(image_dir, video_path, fps=24, image_ext='png'):
    '''
    #codec = cv2.VideoWriter_fourcc(*'XVID')
    #codec = cv2.VideoWriter_fourcc('A','V','C','1')
    #codec = cv2.VideoWriter_fourcc('Y','U','V','1')
    #codec = cv2.VideoWriter_fourcc('P','I','M','1')
    #codec = cv2.VideoWriter_fourcc('M','J','P','G')
    codec = cv2.VideoWriter_fourcc('M','P','4','2')
    #codec = cv2.VideoWriter_fourcc('D','I','V','3')
    #codec =  cv2.VideoWriter_fourcc('D','I','V','X')
    #codec = cv2.VideoWriter_fourcc('U','2','6','3')
    #codec = cv2.VideoWriter_fourcc('I','2','6','3')
    #codec = cv2.VideoWriter_fourcc('F','L','V','1')
    #codec = cv2.VideoWriter_fourcc('H','2','6','4')
    #codec = cv2.VideoWriter_fourcc('A','Y','U','V')
    #codec = cv2.VideoWriter_fourcc('I','U','Y','V')
    编码器常用的几种：
    cv2.VideoWriter_fourcc("I", "4", "2", "0") 
        压缩的yuv颜色编码器，4:2:0色彩度子采样 兼容性好，产生很大的视频 avi
    cv2.VideoWriter_fourcc("P", I", "M", "1")
        采用mpeg-1编码，文件为avi
    cv2.VideoWriter_fourcc("X", "V", "T", "D")
        采用mpeg-4编码，得到视频大小平均 拓展名avi
    cv2.VideoWriter_fourcc("T", "H", "E", "O")
        Ogg Vorbis， 拓展名为ogv
    cv2.VideoWriter_fourcc("F", "L", "V", "1")
        FLASH视频，拓展名为.flv
    '''
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.{}'.format(image_ext))))
    print(len(image_files))
    height, width, _ = cv2.imread(image_files[0]).shape
    out_fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # cv2.VideoWriter_fourcc(*'MP4V')
    out_video = cv2.VideoWriter(video_path, out_fourcc, fps, (width, height))

    for image_file in image_files:
        img = cv2.imread(image_file)
        img = cv2.resize(img, (width, height), interpolation=3)
        out_video.write(img)
    out_video.release()


def add_video_compression(imgs):
    codec_type = ['libx264', 'h264', 'mpeg4']
    codec_prob = [1 / 3., 1 / 3., 1 / 3.]
    codec = random.choices(codec_type, codec_prob)[0]
    # codec = 'mpeg4'
    bitrate = [1e4, 1e5]
    bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)

    buf = io.BytesIO()
    with av.open(buf, 'w', 'mp4') as container:
        stream = container.add_stream(codec, rate=1)
        stream.height = imgs[0].shape[0]
        stream.width = imgs[0].shape[1]
        stream.pix_fmt = 'yuv420p'
        stream.bit_rate = bitrate
        
        for img in imgs:
            img = np.uint8((img.clip(0, 1)*255.).round())
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            frame.pict_type = 'NONE'
            # pdb.set_trace()
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

    outputs = []
    with av.open(buf, 'r', 'mp4') as container:
        if container.streams.video:
            for frame in container.decode(**{'video': 0}):
                outputs.append(
                    frame.to_rgb().to_ndarray().astype(np.float32) / 255.)

    #outputs = np.stack(outputs, axis=0)
    return outputs


if __name__ == '__main__':

    # -----------------------------------
    # test VideoReader(filename, cache_capacity=10)
    # -----------------------------------
#    video_reader = VideoReader('utils/test.mp4')
#    from utils import utils_image as util
#    inputs = []
#    for frame in video_reader:
#        print(frame.dtype)
#        util.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#        #util.imshow(np.flip(frame, axis=2))

    # -----------------------------------
    # test video2images(video_path, output_dir)
    # -----------------------------------
#    video2images('utils/test.mp4', 'frames')

    # -----------------------------------
    # test images2video(image_dir, video_path, fps=24, image_ext='png')
    # -----------------------------------
#    images2video('frames', 'video_02.mp4', fps=30, image_ext='png')


    # -----------------------------------
    # test frames2video(frame_dir, video_file, fps=30, fourcc='XVID', filename_tmpl='{:06d}.png')
    # -----------------------------------
#    frames2video('frames', 'video_01.mp4', filename_tmpl='{:06d}.png')


    # -----------------------------------
    # test add_video_compression(imgs)
    # -----------------------------------
#    imgs = []
#    image_ext = 'png'
#    frames = 'frames'
#    from utils import utils_image as util
#    image_files = sorted(glob.glob(os.path.join(frames, '*.{}'.format(image_ext))))
#    for i, image_file in enumerate(image_files):
#        if i < 7:
#            img = util.imread_uint(image_file, 3)
#            img = util.uint2single(img)
#            imgs.append(img)
#
#    results = add_video_compression(imgs)
#    for i, img in enumerate(results):
#        util.imshow(util.single2uint(img))
#        util.imsave(util.single2uint(img),f'{i:05}.png')

    # run utils/utils_video.py







