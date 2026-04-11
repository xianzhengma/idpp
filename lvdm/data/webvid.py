import os
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import ipdb
import open_clip
import json
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class WebVid(Dataset):

    def __init__(self, meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 512], frame_stride=1, frame_stride_min=1, spatial_transform=None, crop_resolution=None, fps_max=None, load_raw_resolution=False, fixed_fps=None, random_fs=False):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.split_file = json.load(open('/path/to/Genesis/train_test_random_split_0_9000_2.11.json'))
        self.data_dir = '/path/to/Genesis/mpm_bounce_ball_videos6'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4') and int(f.split('_')[-6]) < 9000 and (int(f.split('_')[-6]) in self.split_file['train'])]
        self.metadata.sort()
        spatial_transform = 'resize'
        self.gt_ann_file = json.load(open('/path/to/Genesis/mpm_bounce_ball_videos6/gt_retitution.json'))
        if spatial_transform is not None:
            if spatial_transform == 'random_crop':
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == 'center_crop':
                self.spatial_transform = transforms.Compose([transforms.CenterCrop(resolution)])
            elif spatial_transform == 'resize_center_crop':
                self.spatial_transform = transforms.Compose([transforms.Resize(min(self.resolution)), transforms.CenterCrop(self.resolution)])
            elif spatial_transform == 'resize':
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = os.path.join(self.data_dir, sample)
            caption = ''
            mask_path = video_path.replace('_rgb.mp4', '_seg.mp4')
            if not self.load_raw_resolution:
                ipdb.set_trace()
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                    mask_reader = VideoReader(mask_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                    mask_reader = VideoReader(mask_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f'video length ({len(video_reader)}) is smaller than target length({self.video_length})')
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f'Load video failed! path = {video_path}')
                continue
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))
            frame_stride = max(frame_stride, 1)
            required_frame_num = frame_stride * (self.video_length - 1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            random_range = frame_num - required_frame_num
            start_idx = 0
            frame_stride = 6
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f'Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]')
                index += 1
                continue
        assert frames.shape[0] == self.video_length, f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float()
        masks = masks.mean(dim=0).unsqueeze(0) > 128
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        gt_ann = self.gt_ann_file[video_path.split('_')[-6]]
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': masks, 'gt_ann': gt_ann}
        return data

    def __len__(self):
        return len(self.metadata)

class WebVidTest0(Dataset):

    def __init__(self, meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 512], frame_stride=1, frame_stride_min=1, spatial_transform=None, crop_resolution=None, fps_max=None, load_raw_resolution=False, fixed_fps=None, random_fs=False):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.split_file = json.load(open('/path/to/Genesis/train_test_random_split_0_9000_2.11.json'))
        self.data_dir = '/path/to/Genesis/mpm_bounce_ball_videos6'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4') and int(f.split('_')[-6]) < 9000 and (int(f.split('_')[-6]) in self.split_file['test'])]
        self.metadata.sort()
        spatial_transform = 'resize'
        self.gt_ann_file = json.load(open('/path/to/Genesis/mpm_bounce_ball_videos6/gt_retitution.json'))
        if spatial_transform is not None:
            if spatial_transform == 'random_crop':
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == 'center_crop':
                self.spatial_transform = transforms.Compose([transforms.CenterCrop(resolution)])
            elif spatial_transform == 'resize_center_crop':
                self.spatial_transform = transforms.Compose([transforms.Resize(min(self.resolution)), transforms.CenterCrop(self.resolution)])
            elif spatial_transform == 'resize':
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = os.path.join(self.data_dir, sample)
            caption = ''
            mask_path = video_path.replace('_rgb.mp4', '_seg.mp4')
            if not self.load_raw_resolution:
                ipdb.set_trace()
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                    mask_reader = VideoReader(mask_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                    mask_reader = VideoReader(mask_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f'video length ({len(video_reader)}) is smaller than target length({self.video_length})')
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f'Load video failed! path = {video_path}')
                continue
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))
            frame_stride = max(frame_stride, 1)
            required_frame_num = frame_stride * (self.video_length - 1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            random_range = frame_num - required_frame_num
            start_idx = 0
            frame_stride = 6
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f'Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]')
                index += 1
                continue
        assert frames.shape[0] == self.video_length, f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float()
        masks = masks.mean(dim=0).unsqueeze(0) > 128
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        gt_ann = self.gt_ann_file[video_path.split('_')[-6]]
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': masks, 'gt_ann': gt_ann}
        return data

    def __len__(self):
        return len(self.metadata)

class WebVidTest1(WebVidTest0):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_file = json.load(open('/path/to/Genesis/train_test_random_split_9000_10000_2.11.json'))
        self.data_dir = '/path/to/Genesis/mpm_bounce_ball_videos6'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4') and int(f.split('_')[-6]) in self.split_file['test']]
        self.metadata.sort()
        self.gt_ann_file = json.load(open('/path/to/Genesis/mpm_bounce_ball_videos6/gt_retitution.json'))

class WebVidTest2(WebVidTest0):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_file = json.load(open('/path/to/Genesis/train_test_random_split_9000_10000_2.11.json'))
        self.data_dir = '/path/to/Genesis/mpm_bounce_ball_videos6_test2'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4') and int(f.split('_')[-6]) in self.split_file['test']]
        self.metadata.sort()
        self.gt_ann_file = json.load(open('/path/to/Genesis/mpm_bounce_ball_videos6_test2/gt_retitution.json'))

class WebVidTest3(WebVidTest0):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_file = json.load(open('/path/to/Genesis/train_test_random_split_9000_10000_2.11.json'))
        self.data_dir = '/path/to/Genesis/mpm_bounce_ball_videos6_test3'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4') and int(f.split('_')[-6]) in self.split_file['test']]
        self.metadata.sort()
        self.gt_ann_file = json.load(open('/path/to/Genesis/mpm_bounce_ball_videos6_test3/gt_retitution.json'))

class WebVidTest4(WebVidTest0):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_file = json.load(open('/path/to/Genesis/train_test_random_split_9000_10000_2.11.json'))
        self.data_dir = '/path/to/Genesis/mpm_bounce_ball_videos6_test4'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4') and int(f.split('_')[-6]) in self.split_file['test']]
        self.metadata.sort()
        self.gt_ann_file = json.load(open('/path/to/Genesis/mpm_bounce_ball_videos6_test4/gt_retitution.json'))

class WebVid217v1(Dataset):

    def __init__(self, meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 512], frame_stride=1, frame_stride_min=1, spatial_transform=None, crop_resolution=None, fps_max=None, load_raw_resolution=False, fixed_fps=None, random_fs=False):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        self.metadata.sort()
        spatial_transform = 'resize'
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution.json')))
        if spatial_transform is not None:
            if spatial_transform == 'random_crop':
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == 'center_crop':
                self.spatial_transform = transforms.Compose([transforms.CenterCrop(resolution)])
            elif spatial_transform == 'resize_center_crop':
                self.spatial_transform = transforms.Compose([transforms.Resize(min(self.resolution)), transforms.CenterCrop(self.resolution)])
            elif spatial_transform == 'resize':
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = os.path.join(self.data_dir, sample)
            caption = ''
            mask_path = video_path.replace('_rgb.mp4', '_seg.mp4')
            if not self.load_raw_resolution:
                ipdb.set_trace()
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                    mask_reader = VideoReader(mask_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                    mask_reader = VideoReader(mask_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f'video length ({len(video_reader)}) is smaller than target length({self.video_length})')
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f'Load video failed! path = {video_path}')
                continue
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))
            frame_stride = max(frame_stride, 1)
            required_frame_num = frame_stride * (self.video_length - 1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            random_range = frame_num - required_frame_num
            start_idx = 0
            frame_stride = self.frame_stride
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f'Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]')
                index += 1
                continue
        assert frames.shape[0] == self.video_length, f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float()
        masks = masks.mean(dim=0).unsqueeze(0) > 128
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': masks, 'gt_ann': gt_ann}
        return data

    def __len__(self):
        return len(self.metadata)

class WebVid217v1Test3(WebVid217v1):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = '/path/to/Genesis/real_elastic_benchmark_2.20'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        self.metadata.sort()
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution.json')))

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = os.path.join(self.data_dir, sample)
            caption = ''
            mask_path = video_path.replace('_rgb.mp4', '_seg.mp4')
            if not self.load_raw_resolution:
                ipdb.set_trace()
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                    mask_reader = VideoReader(mask_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                    mask_reader = VideoReader(mask_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f'video length ({len(video_reader)}) is smaller than target length({self.video_length})')
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f'Load video failed! path = {video_path}')
                continue
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))
            frame_stride = max(frame_stride, 1)
            required_frame_num = frame_stride * (self.video_length - 1) + 1
            frame_num = len(video_reader)
            if frame_num < self.video_length:
                print(f'Get frames failed! path = {video_path}; ')
                index += 1
                continue
            start_idx = 0
            frame_stride = frame_num // self.video_length
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f'Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]')
                index += 1
                continue
        assert frames.shape[0] == self.video_length, f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float()
        masks = masks.mean(dim=0).unsqueeze(0) > 128
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': masks, 'gt_ann': gt_ann}
        return data

class WebVid217v1Test1(WebVid217v1):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = '/path/to/shared/genesis_data/2.16_mpm_bounce_ball_videos_3_test1'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        self.metadata.sort()
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution.json')))

class WebVid217v1Test2(WebVid217v1):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = '/path/to/shared/genesis_data/2.16_mpm_bounce_ball_videos_3_test2'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        self.metadata.sort()
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution.json')))

class WebVid217v1Enhanced225(WebVid217v1):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = '/path/to/shared/genesis_data/2.16_mpm_bounce_ball_videos_4_incorporated_enhanced_2'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        self.metadata.sort()
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution.json')))

class WebVid408v1(Dataset):

    def __init__(self, meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 512], frame_stride=1, frame_stride_min=1, spatial_transform=None, crop_resolution=None, fps_max=None, load_raw_resolution=False, fixed_fps=None, random_fs=False):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        spatial_transform = 'resize'
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution2.json')))
        self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys()]
        self.metadata.sort()
        if spatial_transform is not None:
            if spatial_transform == 'random_crop':
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == 'center_crop':
                self.spatial_transform = transforms.Compose([transforms.CenterCrop(resolution)])
            elif spatial_transform == 'resize_center_crop':
                self.spatial_transform = transforms.Compose([transforms.Resize(min(self.resolution)), transforms.CenterCrop(self.resolution)])
            elif spatial_transform == 'resize':
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = os.path.join(self.data_dir, sample)
            caption = ''
            mask_path = video_path.replace('_rgb.mp4', '_seg.mp4')
            if not self.load_raw_resolution:
                ipdb.set_trace()
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                    mask_reader = VideoReader(mask_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                    mask_reader = VideoReader(mask_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f'video length ({len(video_reader)}) is smaller than target length({self.video_length})')
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f'Load video failed! path = {video_path}')
                continue
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))
            frame_stride = max(frame_stride, 1)
            required_frame_num = frame_stride * (self.video_length - 1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            random_range = frame_num - required_frame_num
            start_idx = 0
            frame_stride = self.frame_stride
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f'Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]')
                index += 1
                continue
        assert frames.shape[0] == self.video_length, f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float()
        masks = masks.mean(dim=0).unsqueeze(0) > 128
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': masks, 'gt_ann': gt_ann}
        return data

    def __len__(self):
        return len(self.metadata)

class WebVid408v1debug(Dataset):

    def __init__(self, meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 512], frame_stride=1, frame_stride_min=1, spatial_transform=None, crop_resolution=None, fps_max=None, load_raw_resolution=False, fixed_fps=None, random_fs=False):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        spatial_transform = 'resize'
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution2.json')))
        self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys() if float(f.split('_')[3]) < 0.03]
        self.metadata.sort()
        if spatial_transform is not None:
            if spatial_transform == 'random_crop':
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == 'center_crop':
                self.spatial_transform = transforms.Compose([transforms.CenterCrop(resolution)])
            elif spatial_transform == 'resize_center_crop':
                self.spatial_transform = transforms.Compose([transforms.Resize(min(self.resolution)), transforms.CenterCrop(self.resolution)])
            elif spatial_transform == 'resize':
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = os.path.join(self.data_dir, sample)
            caption = ''
            mask_path = video_path.replace('_rgb.mp4', '_seg.mp4')
            if not self.load_raw_resolution:
                ipdb.set_trace()
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                    mask_reader = VideoReader(mask_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                    mask_reader = VideoReader(mask_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f'video length ({len(video_reader)}) is smaller than target length({self.video_length})')
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f'Load video failed! path = {video_path}')
                continue
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))
            frame_stride = max(frame_stride, 1)
            required_frame_num = frame_stride * (self.video_length - 1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            random_range = frame_num - required_frame_num
            start_idx = 0
            frame_stride = self.frame_stride
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f'Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]')
                index += 1
                continue
        assert frames.shape[0] == self.video_length, f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float()
        masks = masks.mean(dim=0).unsqueeze(0) > 128
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': masks, 'gt_ann': gt_ann}
        return data

    def __len__(self):
        return len(self.metadata)

class WebVidv2Elasticity(Dataset):

    def __init__(self, meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 512], frame_stride=1, frame_stride_min=1, spatial_transform=None, crop_resolution=None, fps_max=None, load_raw_resolution=False, fixed_fps=None, random_fs=False):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        spatial_transform = 'resize'
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'bounce_analysis_results.json')))
        self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys()]
        self.metadata.sort()
        if spatial_transform is not None:
            if spatial_transform == 'random_crop':
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == 'center_crop':
                self.spatial_transform = transforms.Compose([transforms.CenterCrop(resolution)])
            elif spatial_transform == 'resize_center_crop':
                self.spatial_transform = transforms.Compose([transforms.Resize(min(self.resolution)), transforms.CenterCrop(self.resolution)])
            elif spatial_transform == 'resize':
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = os.path.join(self.data_dir, sample)
            caption = ''
            mask_path = video_path.replace('_rgb.mp4', '_seg.mp4')
            if not self.load_raw_resolution:
                ipdb.set_trace()
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                    mask_reader = VideoReader(mask_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                    mask_reader = VideoReader(mask_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f'video length ({len(video_reader)}) is smaller than target length({self.video_length})')
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f'Load video failed! path = {video_path}')
                continue
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))
            frame_stride = max(frame_stride, 1)
            required_frame_num = frame_stride * (self.video_length - 1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            random_range = frame_num - required_frame_num
            start_idx = 0
            frame_stride = self.frame_stride
            gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            key_timestep_indices = gt_ann['temporal_key_points']
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                keytime_masks = mask_reader.get_batch(key_timestep_indices)
                break
            except:
                print(f'Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]')
                index += 1
                continue
        assert frames.shape[0] == self.video_length, f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float()
        keytime_masks = torch.tensor(keytime_masks.asnumpy()).permute(3, 0, 1, 2).float()
        masks = masks.mean(dim=0).unsqueeze(0) > 128
        keytime_masks = keytime_masks.mean(dim=0).unsqueeze(0) > 128
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
            keytime_masks = self.spatial_transform(keytime_masks)
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': masks, 'keytime_mask': keytime_masks, 'gt_ann': gt_ann}
        return data

    def __len__(self):
        return len(self.metadata)

class WebVidv2Viscosity(Dataset):

    def __init__(self, meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 512], frame_stride=1, frame_stride_min=1, spatial_transform=None, crop_resolution=None, fps_max=None, load_raw_resolution=False, fixed_fps=None, random_fs=False, red_circle=False, red_circle2=False, red_circle2_vjepa=False, meta_name=None, video_list_json=None):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.red_circle = red_circle
        self.red_circle2 = red_circle2
        self.red_circle2_vjepa = red_circle2_vjepa
        self.meta_name = meta_name
        self.video_list_json = video_list_json
        spatial_transform = 'resize'
        if self.meta_name is None:
            self.meta_name = 'bounce_analysis_results.json'
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, self.meta_name)))
        self._no_red_eval = not self.red_circle and (not self.red_circle2) and (not self.red_circle2_vjepa)
        if self._no_red_eval:
            self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys() if os.path.exists(os.path.join(self.data_dir, f + '_rgb.mp4'))]
        else:
            self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys() if os.path.exists(os.path.join(self.data_dir, f + '_rgb.mp4')) and os.path.exists(os.path.join(self.data_dir, f + '_seg.mp4'))]
        self.metadata.sort()
        if self.video_list_json is not None:
            try:
                selected_entries = json.load(open(self.video_list_json, 'r'))
                selected_video_names = []
                for entry in selected_entries:
                    if isinstance(entry, str):
                        selected_video_names.append(entry + '_rgb.mp4')
                    elif isinstance(entry, (list, tuple)) and len(entry) > 0:
                        selected_video_names.append(str(entry[0]) + '_rgb.mp4')
                selected_set = set(selected_video_names)
                self.metadata = [name for name in self.metadata if name in selected_set]
            except (OSError, ValueError, TypeError):
                print(f'WARNING: failed to load video_list_json={self.video_list_json}, fallback to full metadata')
        print(len(self.metadata))
        if spatial_transform is not None:
            if spatial_transform == 'random_crop':
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == 'center_crop':
                self.spatial_transform = transforms.Compose([transforms.CenterCrop(resolution)])
            elif spatial_transform == 'resize_center_crop':
                self.spatial_transform = transforms.Compose([transforms.Resize(min(self.resolution)), transforms.CenterCrop(self.resolution)])
            elif spatial_transform == 'resize':
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = os.path.join(self.data_dir, sample)
            caption = ''
            if self.red_circle == False and self.red_circle2 == False:
                mask_path = video_path.replace('_rgb.mp4', '_seg.mp4')
            elif self.red_circle:
                mask_path = video_path.replace('_rgb.mp4', '_red.mp4')
            else:
                assert self.red_circle2
                mask_path = video_path.replace('_rgb.mp4', '_red2.mp4')
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                mask_reader = None
                if not self._no_red_eval:
                    if self.load_raw_resolution:
                        mask_reader = VideoReader(mask_path, ctx=cpu(0))
                    else:
                        mask_reader = VideoReader(mask_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f'video length ({len(video_reader)}) is smaller than target length({self.video_length})')
                    index += 1
                    continue
                else:
                    pass
            except Exception:
                index += 1
                print(f'Load video failed! path = {video_path}')
                continue
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))
            frame_stride = max(frame_stride, 1)
            required_frame_num = frame_stride * (self.video_length - 1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            random_range = frame_num - required_frame_num
            start_idx = 0
            frame_stride = self.frame_stride
            gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                if not self._no_red_eval:
                    masks = mask_reader.get_batch(frame_indices)
                break
            except Exception:
                print(f'Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]')
                index += 1
                continue
        assert frames.shape[0] == self.video_length, f'{len(frames)}, self.video_length={self.video_length}'
        fr_np = frames.asnumpy()
        frames = torch.tensor(fr_np).permute(3, 0, 1, 2).float()
        if self._no_red_eval:
            mk_np = np.full(fr_np.shape, 255, dtype=fr_np.dtype)
        else:
            mk_np = masks.asnumpy()
        masks = torch.tensor(mk_np).permute(3, 0, 1, 2).float()
        if self.red_circle == False and self.red_circle2 == False and (self.red_circle2_vjepa == False):
            masks = masks.mean(dim=0).unsqueeze(0) > 128
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        if self.red_circle2_vjepa:
            masks = masks / 255.0
            for c in range(min(masks.shape[0], 3)):
                masks[c] = (masks[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': masks, 'keytime_mask': masks, 'gt_ann': gt_ann}
        return data

    def __len__(self):
        return len(self.metadata)

class WebVidv3Viscosity(Dataset):

    def __init__(self, meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 512], frame_stride=1, frame_stride_min=1, spatial_transform=None, crop_resolution=None, fps_max=None, load_raw_resolution=False, fixed_fps=None, random_fs=False, red_circle=False, red_circle2=False, red_circle2_vjepa=False, video_list_json=None):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.red_circle = red_circle
        self.red_circle2 = red_circle2
        self.red_circle2_vjepa = red_circle2_vjepa
        self.video_list_json = video_list_json
        self.video_groups = {}
        self.all_basenames = []
        all_files = os.listdir(self.data_dir)
        rgb_files = [f for f in all_files if f.endswith('_rgb.mp4')]
        if self.red_circle2 or self.red_circle2_vjepa:
            mask_pair_name = '_red2.mp4'
        elif self.red_circle:
            mask_pair_name = '_red.mp4'
        else:
            mask_pair_name = '_seg.mp4'
        for rgb_file in rgb_files:
            basename = rgb_file[:-8]
            mask_file = basename + mask_pair_name
            if mask_file in all_files:
                parts = basename.split('_')
                if len(parts) >= 3 and parts[0] == 'video':
                    try:
                        video_i = int(parts[1])
                        if video_i not in self.video_groups:
                            self.video_groups[video_i] = []
                        self.video_groups[video_i].append(basename)
                        self.all_basenames.append(basename)
                    except ValueError:
                        continue
        self.valid_video_groups = {k: v for k, v in self.video_groups.items() if len(v) >= 2}
        self.metadata = list(self.valid_video_groups.keys())
        print(f'Found {len(self.metadata)} video groups with multiple videos for comparison')
        if self.video_list_json is not None:
            try:
                selected_entries = json.load(open(self.video_list_json, 'r'))
                selected_pairs = []
                all_basename_set = set(self.all_basenames)
                for entry in selected_entries:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        b1 = str(entry[0])
                        b2 = str(entry[1])
                        if b1 in all_basename_set and b2 in all_basename_set:
                            selected_pairs.append([b1, b2])
                if len(selected_pairs) > 0:
                    self.metadata = selected_pairs
                    print(f'Using explicit pair list from {self.video_list_json}, total pairs: {len(self.metadata)}')
                else:
                    print(f'WARNING: video_list_json={self.video_list_json} matched 0 pairs in data_dir; keeping grouped video_i metadata (train sampler).')
            except (OSError, ValueError, TypeError):
                print(f'WARNING: failed to load video_list_json={self.video_list_json}, fallback to grouped metadata')
        if spatial_transform is not None:
            if spatial_transform == 'random_crop':
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == 'center_crop':
                self.spatial_transform = transforms.Compose([transforms.CenterCrop(resolution)])
            elif spatial_transform == 'resize_center_crop':
                self.spatial_transform = transforms.Compose([transforms.Resize(min(self.resolution)), transforms.CenterCrop(self.resolution)])
            elif spatial_transform == 'resize':
                self.spatial_transform = transforms.Resize(self.resolution)
            elif spatial_transform == 'extreme_augmentation':
                self.spatial_transform = transforms.Compose([transforms.RandomResizedCrop(size=self.resolution, scale=(0.8, 1.0), ratio=(1.4, 1.6)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.3), transforms.RandomRotation(degrees=30), transforms.RandomPerspective(distortion_scale=0.3, p=0.5), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), transforms.RandomGrayscale(p=0.1), transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))])
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def _get_property_value(self, basename):
        try:
            return float(basename.split('_')[7])
        except (IndexError, ValueError):
            return 0.0

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            meta_entry = self.metadata[index]
            explicit_pair_mode = isinstance(meta_entry, (list, tuple)) and len(meta_entry) >= 2
            if explicit_pair_mode:
                video_basename1 = str(meta_entry[0])
                video_basename2 = str(meta_entry[1])
                if video_basename1 not in self.all_basenames or video_basename2 not in self.all_basenames:
                    index += 1
                    continue
            else:
                video_i = meta_entry
                available_basenames = self.valid_video_groups[video_i]
                if len(available_basenames) < 2:
                    index += 1
                    continue
                sampled_basenames = random.sample(available_basenames, 2)
                video_basename1 = sampled_basenames[0]
                video_basename2 = sampled_basenames[1]
            prop_value1 = self._get_property_value(video_basename1)
            prop_value2 = self._get_property_value(video_basename2)
            if not explicit_pair_mode and prop_value1 != 0 and (prop_value2 != 0):
                ratio = max(prop_value1, prop_value2) / min(prop_value1, prop_value2)
                if ratio < 1.2:
                    index += 1
                    continue
            video_path1 = os.path.join(self.data_dir, video_basename1 + '_rgb.mp4')
            video_path2 = os.path.join(self.data_dir, video_basename2 + '_rgb.mp4')
            if self.red_circle == False and self.red_circle2 == False:
                mask_path1 = os.path.join(self.data_dir, video_basename1 + '_seg.mp4')
                mask_path2 = os.path.join(self.data_dir, video_basename2 + '_seg.mp4')
            elif self.red_circle:
                mask_path1 = os.path.join(self.data_dir, video_basename1 + '_red.mp4')
                mask_path2 = os.path.join(self.data_dir, video_basename2 + '_red.mp4')
            else:
                assert self.red_circle2
                mask_path1 = os.path.join(self.data_dir, video_basename1 + '_red2.mp4')
                mask_path2 = os.path.join(self.data_dir, video_basename2 + '_red2.mp4')
            caption = ''
            try:
                if self.load_raw_resolution:
                    video_reader1 = VideoReader(video_path1, ctx=cpu(0))
                    mask_reader1 = VideoReader(mask_path1, ctx=cpu(0))
                    video_reader2 = VideoReader(video_path2, ctx=cpu(0))
                    mask_reader2 = VideoReader(mask_path2, ctx=cpu(0))
                else:
                    video_reader1 = VideoReader(video_path1, ctx=cpu(0), width=530, height=300)
                    mask_reader1 = VideoReader(mask_path1, ctx=cpu(0), width=530, height=300)
                    video_reader2 = VideoReader(video_path2, ctx=cpu(0), width=530, height=300)
                    mask_reader2 = VideoReader(mask_path2, ctx=cpu(0), width=530, height=300)
                if len(video_reader1) < self.video_length or len(video_reader2) < self.video_length:
                    print(f'Video length too small: {len(video_reader1)}, {len(video_reader2)} < {self.video_length}')
                    index += 1
                    continue
            except Exception as e:
                index += 1
                print(f'Load video failed! paths = {video_path1}, {video_path2}, error: {e}')
                continue
            fps_ori = video_reader1.get_avg_fps()
            start_idx = 0
            frame_stride = self.frame_stride
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            try:
                frames1 = video_reader1.get_batch(frame_indices)
                masks1 = mask_reader1.get_batch(frame_indices)
                frames2 = video_reader2.get_batch(frame_indices)
                masks2 = mask_reader2.get_batch(frame_indices)
                break
            except Exception as e:
                print(f'Get frames failed! paths = {video_path1}, {video_path2}, error: {e}')
                index += 1
                continue
        assert frames1.shape[0] == self.video_length and frames2.shape[0] == self.video_length, f'Frame lengths: {frames1.shape[0]}, {frames2.shape[0]}, expected: {self.video_length}'
        frames1 = torch.tensor(frames1.asnumpy()).permute(3, 0, 1, 2).float()
        masks1 = torch.tensor(masks1.asnumpy()).permute(3, 0, 1, 2).float()
        frames2 = torch.tensor(frames2.asnumpy()).permute(3, 0, 1, 2).float()
        masks2 = torch.tensor(masks2.asnumpy()).permute(3, 0, 1, 2).float()
        if self.red_circle == False and self.red_circle2 == False and (self.red_circle2_vjepa == False):
            masks1 = masks1.mean(dim=0).unsqueeze(0) > 128
            masks2 = masks2.mean(dim=0).unsqueeze(0) > 128
        if self.spatial_transform is not None:
            if self.spatial_transform == 'extreme_augmentation':
                frames1 = self.spatial_transform(frames1.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
                masks1 = self.spatial_transform(masks1.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
                frames2 = self.spatial_transform(frames2.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
                masks2 = self.spatial_transform(masks2.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
            else:
                frames1 = self.spatial_transform(frames1)
                masks1 = self.spatial_transform(masks1)
                frames2 = self.spatial_transform(frames2)
                masks2 = self.spatial_transform(masks2)
        if self.resolution is not None:
            assert (frames1.shape[2], frames1.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames1={frames1.shape}, self.resolution={self.resolution}'
            assert (frames2.shape[2], frames2.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames2={frames2.shape}, self.resolution={self.resolution}'
        frames1 = (frames1 / 255 - 0.5) * 2
        frames2 = (frames2 / 255 - 0.5) * 2
        if self.red_circle or self.red_circle2:
            masks1 = (masks1 / 255 - 0.5) * 2
            masks2 = (masks2 / 255 - 0.5) * 2
        elif self.red_circle2_vjepa:
            masks1 = masks1 / 255.0
            masks2 = masks2 / 255.0
            for c in range(min(masks1.shape[0], 3)):
                masks1[c] = (masks1[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]
                masks2[c] = (masks2[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]
        frames = torch.stack([frames1, frames2], dim=0)
        masks = torch.stack([masks1, masks2], dim=0)
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        gt_restitution = 1 if prop_value1 > prop_value2 else 0
        gt_ann = {'gt_restitution': gt_restitution, 'prop_value1': prop_value1, 'prop_value2': prop_value2, 'video_basename1': video_basename1, 'video_basename2': video_basename2}
        data = {'video': frames, 'caption': caption, 'path': video_path1, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': masks, 'keytime_mask': masks, 'gt_ann': gt_ann}
        return data

    def __len__(self):
        return len(self.metadata)

class WebVidv3ViscosityTest(Dataset):

    def __init__(self, meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 512], frame_stride=1, frame_stride_min=1, spatial_transform=None, crop_resolution=None, fps_max=None, load_raw_resolution=False, fixed_fps=None, random_fs=False, red_circle=False, red_circle2=False, red_circle2_vjepa=False, video_list_json=None):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.red_circle = red_circle
        self.red_circle2 = red_circle2
        self.red_circle2_vjepa = red_circle2_vjepa
        self.video_list_json = video_list_json
        spatial_transform = 'resize'
        self.video_groups = {}
        self.all_basenames = []
        all_files = os.listdir(self.data_dir)
        all_files.sort()
        rgb_files = [f for f in all_files if f.endswith('_rgb.mp4')]
        if self.red_circle2 or self.red_circle2_vjepa:
            mask_pair_name = '_red2.mp4'
        elif self.red_circle:
            mask_pair_name = '_red.mp4'
        else:
            mask_pair_name = '_seg.mp4'
        for rgb_file in rgb_files:
            basename = rgb_file[:-8]
            mask_file = basename + mask_pair_name
            if mask_file in all_files:
                parts = basename.split('_')
                if len(parts) >= 3 and parts[0] == 'video':
                    try:
                        video_i = int(parts[1])
                        if video_i not in self.video_groups:
                            self.video_groups[video_i] = []
                        self.video_groups[video_i].append(basename)
                        self.all_basenames.append(basename)
                    except ValueError:
                        continue
        self.valid_video_groups = {k: v for k, v in self.video_groups.items() if len(v) >= 2}
        self.metadata = list(self.valid_video_groups.keys())
        print(f'Found {len(self.metadata)} video groups with multiple videos for comparison')
        if self.video_list_json is not None:
            try:
                selected_entries = json.load(open(self.video_list_json, 'r'))
                selected_pairs = []
                all_basename_set = set(self.all_basenames)
                for entry in selected_entries:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        b1 = str(entry[0])
                        b2 = str(entry[1])
                        if b1 in all_basename_set and b2 in all_basename_set:
                            selected_pairs.append([b1, b2])
                self.metadata = selected_pairs
                print(f'Using explicit pair list from {self.video_list_json}, total pairs: {len(self.metadata)}')
            except (OSError, ValueError, TypeError):
                print(f'WARNING: failed to load video_list_json={self.video_list_json}, fallback to grouped metadata')
        if spatial_transform is not None:
            if spatial_transform == 'random_crop':
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == 'center_crop':
                self.spatial_transform = transforms.Compose([transforms.CenterCrop(resolution)])
            elif spatial_transform == 'resize_center_crop':
                self.spatial_transform = transforms.Compose([transforms.Resize(min(self.resolution)), transforms.CenterCrop(self.resolution)])
            elif spatial_transform == 'resize':
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def _get_property_value(self, basename):
        try:
            return float(basename.split('_')[7])
        except (IndexError, ValueError):
            return 0.0

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            meta_entry = self.metadata[index]
            explicit_pair_mode = isinstance(meta_entry, (list, tuple)) and len(meta_entry) >= 2
            if explicit_pair_mode:
                video_basename1 = str(meta_entry[0])
                video_basename2 = str(meta_entry[1])
                if video_basename1 not in self.all_basenames or video_basename2 not in self.all_basenames:
                    index += 1
                    continue
            else:
                video_i = meta_entry
                available_basenames = self.valid_video_groups[video_i]
                if len(available_basenames) < 2:
                    index += 1
                    continue
                video_basename1 = available_basenames[0]
                video_basename2 = available_basenames[1]
            prop_value1 = self._get_property_value(video_basename1)
            prop_value2 = self._get_property_value(video_basename2)
            if not explicit_pair_mode and prop_value1 != 0 and (prop_value2 != 0):
                ratio = max(prop_value1, prop_value2) / min(prop_value1, prop_value2)
                if ratio < 1.2:
                    index += 1
                    continue
            video_path1 = os.path.join(self.data_dir, video_basename1 + '_rgb.mp4')
            video_path2 = os.path.join(self.data_dir, video_basename2 + '_rgb.mp4')
            if self.red_circle == False and self.red_circle2 == False:
                mask_path1 = os.path.join(self.data_dir, video_basename1 + '_seg.mp4')
                mask_path2 = os.path.join(self.data_dir, video_basename2 + '_seg.mp4')
            elif self.red_circle:
                mask_path1 = os.path.join(self.data_dir, video_basename1 + '_red.mp4')
                mask_path2 = os.path.join(self.data_dir, video_basename2 + '_red.mp4')
            else:
                assert self.red_circle2
                mask_path1 = os.path.join(self.data_dir, video_basename1 + '_red2.mp4')
                mask_path2 = os.path.join(self.data_dir, video_basename2 + '_red2.mp4')
            caption = ''
            try:
                if self.load_raw_resolution:
                    video_reader1 = VideoReader(video_path1, ctx=cpu(0))
                    mask_reader1 = VideoReader(mask_path1, ctx=cpu(0))
                    video_reader2 = VideoReader(video_path2, ctx=cpu(0))
                    mask_reader2 = VideoReader(mask_path2, ctx=cpu(0))
                else:
                    video_reader1 = VideoReader(video_path1, ctx=cpu(0), width=530, height=300)
                    mask_reader1 = VideoReader(mask_path1, ctx=cpu(0), width=530, height=300)
                    video_reader2 = VideoReader(video_path2, ctx=cpu(0), width=530, height=300)
                    mask_reader2 = VideoReader(mask_path2, ctx=cpu(0), width=530, height=300)
                if len(video_reader1) < self.video_length or len(video_reader2) < self.video_length:
                    print(f'Video length too small: {len(video_reader1)}, {len(video_reader2)} < {self.video_length}')
                    index += 1
                    continue
            except Exception as e:
                index += 1
                print(f'Load video failed! paths = {video_path1}, {video_path2}, error: {e}')
                continue
            fps_ori = video_reader1.get_avg_fps()
            start_idx = 0
            frame_stride = self.frame_stride
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            try:
                frames1 = video_reader1.get_batch(frame_indices)
                masks1 = mask_reader1.get_batch(frame_indices)
                frames2 = video_reader2.get_batch(frame_indices)
                masks2 = mask_reader2.get_batch(frame_indices)
                break
            except Exception as e:
                print(f'Get frames failed! paths = {video_path1}, {video_path2}, error: {e}')
                index += 1
                continue
        assert frames1.shape[0] == self.video_length and frames2.shape[0] == self.video_length, f'Frame lengths: {frames1.shape[0]}, {frames2.shape[0]}, expected: {self.video_length}'
        frames1 = torch.tensor(frames1.asnumpy()).permute(3, 0, 1, 2).float()
        masks1 = torch.tensor(masks1.asnumpy()).permute(3, 0, 1, 2).float()
        frames2 = torch.tensor(frames2.asnumpy()).permute(3, 0, 1, 2).float()
        masks2 = torch.tensor(masks2.asnumpy()).permute(3, 0, 1, 2).float()
        if self.red_circle == False and self.red_circle2 == False and (self.red_circle2_vjepa == False):
            masks1 = masks1.mean(dim=0).unsqueeze(0) > 128
            masks2 = masks2.mean(dim=0).unsqueeze(0) > 128
        if self.spatial_transform is not None:
            frames1 = self.spatial_transform(frames1)
            masks1 = self.spatial_transform(masks1)
            frames2 = self.spatial_transform(frames2)
            masks2 = self.spatial_transform(masks2)
        if self.resolution is not None:
            assert (frames1.shape[2], frames1.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames1={frames1.shape}, self.resolution={self.resolution}'
            assert (frames2.shape[2], frames2.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames2={frames2.shape}, self.resolution={self.resolution}'
        frames1 = (frames1 / 255 - 0.5) * 2
        frames2 = (frames2 / 255 - 0.5) * 2
        if self.red_circle2_vjepa:
            masks1 = masks1 / 255.0
            masks2 = masks2 / 255.0
            for c in range(min(masks1.shape[0], 3)):
                masks1[c] = (masks1[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]
                masks2[c] = (masks2[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]
        frames = torch.stack([frames1, frames2], dim=0)
        masks = torch.stack([masks1, masks2], dim=0)
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        gt_restitution = 1 if prop_value1 > prop_value2 else 0
        gt_ann = {'gt_restitution': gt_restitution, 'prop_value1': prop_value1, 'prop_value2': prop_value2, 'video_basename1': video_basename1, 'video_basename2': video_basename2}
        save_name_path = '_'.join(os.path.basename(video_path1).split('_')[:4]) + '--' + '_'.join(os.path.basename(video_path2).split('_')[:4])
        data = {'video': frames, 'caption': caption, 'path': save_name_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': masks, 'keytime_mask': masks, 'gt_ann': gt_ann}
        return data

    def __len__(self):
        return len(self.metadata)

class WebVidv3Friction(WebVidv3Viscosity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_property_value(self, basename):
        try:
            return float(basename.split('_')[8])
        except (IndexError, ValueError):
            return 0.0

class WebVidv3Elasticity(WebVidv3Viscosity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution1.json')))

    def _get_property_value(self, basename):
        return self.gt_file[basename]

class WebVidv3FrictionTest(WebVidv3ViscosityTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_property_value(self, basename):
        try:
            return float(basename.split('_')[8])
        except (IndexError, ValueError):
            return 0.0

class WebVidv3ElasticityTest(WebVidv3ViscosityTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_file = json.load(open(os.path.join(self.data_dir, 'gt_restitution1.json')))

    def _get_property_value(self, basename):
        return self.gt_file[basename]

    def __getitem__(self, index):
        return super().__getitem__(index)

class WebVidv3VFRealTest(Dataset):

    def __init__(self, meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 512], frame_stride=1, frame_stride_min=1, spatial_transform=None, crop_resolution=None, fps_max=None, load_raw_resolution=False, fixed_fps=None, random_fs=False):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        spatial_transform = 'resize'
        self.video_groups = {}
        self.all_basenames = []
        all_files = os.listdir(self.data_dir)
        all_files.sort()
        rgb_files = [f for f in all_files if f.endswith('.mp4')]
        for rgb_file in rgb_files:
            basename = rgb_file[:-4]
            parts = basename.split('_')
            if len(parts) >= 3 and parts[0] == 'video':
                try:
                    video_i = int(parts[1])
                    if video_i not in self.video_groups:
                        self.video_groups[video_i] = []
                    self.video_groups[video_i].append(basename)
                    self.all_basenames.append(basename)
                except ValueError:
                    continue
        self.valid_video_groups = {k: v for k, v in self.video_groups.items() if len(v) >= 2}
        self.metadata = list(self.valid_video_groups.keys())
        print(f'Found {len(self.metadata)} video groups with multiple videos for comparison')
        if spatial_transform is not None:
            if spatial_transform == 'random_crop':
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == 'center_crop':
                self.spatial_transform = transforms.Compose([transforms.CenterCrop(resolution)])
            elif spatial_transform == 'resize_center_crop':
                self.spatial_transform = transforms.Compose([transforms.Resize(min(self.resolution)), transforms.CenterCrop(self.resolution)])
            elif spatial_transform == 'resize':
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def _get_property_value(self, basename):
        try:
            return float(basename.split('_')[8])
        except (IndexError, ValueError):
            return 0.0

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            video_i = self.metadata[index]
            available_basenames = self.valid_video_groups[video_i]
            if len(available_basenames) < 2:
                index += 1
                continue
            video_basename1 = available_basenames[0]
            video_basename2 = available_basenames[1]
            prop_value1 = self._get_property_value(video_basename1)
            prop_value2 = self._get_property_value(video_basename2)
            if prop_value1 != 0 and prop_value2 != 0:
                ratio = max(prop_value1, prop_value2) / min(prop_value1, prop_value2)
                if ratio < 1.2:
                    index += 1
                    continue
            video_path1 = os.path.join(self.data_dir, video_basename1 + '.mp4')
            video_path2 = os.path.join(self.data_dir, video_basename2 + '.mp4')
            caption = ''
            try:
                if self.load_raw_resolution:
                    video_reader1 = VideoReader(video_path1, ctx=cpu(0))
                    video_reader2 = VideoReader(video_path2, ctx=cpu(0))
                else:
                    video_reader1 = VideoReader(video_path1, ctx=cpu(0), width=530, height=300)
                    video_reader2 = VideoReader(video_path2, ctx=cpu(0), width=530, height=300)
                if len(video_reader1) < self.video_length or len(video_reader2) < self.video_length:
                    print(f'Video length too small: {len(video_reader1)}, {len(video_reader2)} < {self.video_length}')
                    index += 1
                    continue
            except Exception as e:
                index += 1
                print(f'Load video failed! paths = {video_path1}, {video_path2}, error: {e}')
                continue
            fps_ori = video_reader1.get_avg_fps()
            start_idx = 0
            frame_stride = self.frame_stride
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            try:
                frames1 = video_reader1.get_batch(frame_indices)
                frames2 = video_reader2.get_batch(frame_indices)
                break
            except Exception as e:
                print(f'Get frames failed! paths = {video_path1}, {video_path2}, error: {e}')
                index += 1
                continue
        assert frames1.shape[0] == self.video_length and frames2.shape[0] == self.video_length, f'Frame lengths: {frames1.shape[0]}, {frames2.shape[0]}, expected: {self.video_length}'
        frames1 = torch.tensor(frames1.asnumpy()).permute(3, 0, 1, 2).float()
        frames2 = torch.tensor(frames2.asnumpy()).permute(3, 0, 1, 2).float()
        if self.spatial_transform is not None:
            frames1 = self.spatial_transform(frames1)
            frames2 = self.spatial_transform(frames2)
        if self.resolution is not None:
            assert (frames1.shape[2], frames1.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames1={frames1.shape}, self.resolution={self.resolution}'
            assert (frames2.shape[2], frames2.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames2={frames2.shape}, self.resolution={self.resolution}'
        frames1 = (frames1 / 255 - 0.5) * 2
        frames2 = (frames2 / 255 - 0.5) * 2
        frames = torch.stack([frames1, frames2], dim=0)
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        gt_restitution = 1 if prop_value1 > prop_value2 else 0
        gt_ann = {'gt_restitution': gt_restitution, 'prop_value1': prop_value1, 'prop_value2': prop_value2, 'video_basename1': video_basename1, 'video_basename2': video_basename2}
        save_name_path = '_'.join(os.path.basename(video_path1).split('_')[:4]) + '--' + '_'.join(os.path.basename(video_path2).split('_')[:4])
        data = {'video': frames, 'caption': caption, 'path': save_name_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': frames, 'keytime_mask': frames, 'gt_ann': gt_ann}
        return data

    def __len__(self):
        return len(self.metadata)

class WebVidv2Elasticitydebug(Dataset):

    def __init__(self, meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 512], frame_stride=1, frame_stride_min=1, spatial_transform=None, crop_resolution=None, fps_max=None, load_raw_resolution=False, fixed_fps=None, random_fs=False):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        spatial_transform = 'resize'
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'bounce_analysis_results.json')))
        self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys() if float(f.split('_')[3]) < 0.03]
        self.metadata.sort()
        if spatial_transform is not None:
            if spatial_transform == 'random_crop':
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == 'center_crop':
                self.spatial_transform = transforms.Compose([transforms.CenterCrop(resolution)])
            elif spatial_transform == 'resize_center_crop':
                self.spatial_transform = transforms.Compose([transforms.Resize(min(self.resolution)), transforms.CenterCrop(self.resolution)])
            elif spatial_transform == 'resize':
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while True:
            index = index % len(self.metadata)
            sample = self.metadata[index]
            video_path = os.path.join(self.data_dir, sample)
            caption = ''
            mask_path = video_path.replace('_rgb.mp4', '_seg.mp4')
            if not self.load_raw_resolution:
                ipdb.set_trace()
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                    mask_reader = VideoReader(mask_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                    mask_reader = VideoReader(mask_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f'video length ({len(video_reader)}) is smaller than target length({self.video_length})')
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f'Load video failed! path = {video_path}')
                continue
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))
            frame_stride = max(frame_stride, 1)
            required_frame_num = frame_stride * (self.video_length - 1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            random_range = frame_num - required_frame_num
            start_idx = 0
            frame_stride = self.frame_stride
            gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            key_timestep_indices = gt_ann['temporal_key_points']
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                keytime_masks = mask_reader.get_batch(key_timestep_indices)
                break
            except:
                print(f'Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]')
                index += 1
                continue
        assert frames.shape[0] == self.video_length, f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float()
        keytime_masks = torch.tensor(keytime_masks.asnumpy()).permute(3, 0, 1, 2).float()
        masks = masks.mean(dim=0).unsqueeze(0) > 128
        keytime_masks = keytime_masks.mean(dim=0).unsqueeze(0) > 128
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
            keytime_masks = self.spatial_transform(keytime_masks)
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask': masks, 'keytime_mask': keytime_masks, 'gt_ann': gt_ann}
        return data

    def __len__(self):
        return len(self.metadata)
if __name__ == '__main__':
    meta_path = ''
    data_dir = ''
    save_dir = ''
    dataset = WebVid(meta_path, data_dir, subsample=None, video_length=16, resolution=[256, 448], frame_stride=4, spatial_transform='resize_center_crop', crop_resolution=None, fps_max=None, load_raw_resolution=True)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
    from utils.save_video import tensor_to_mp4
    for i, batch in tqdm(enumerate(dataloader), desc='Data Batch'):
        video = batch['video']
        name = batch['path'][0].split('videos/')[-1].replace('/', '_')
        tensor_to_mp4(video, save_dir + '/' + name, fps=8)
