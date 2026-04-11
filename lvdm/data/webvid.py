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

# VJEPA归一化常量
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class WebVid(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 ):
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
        # self._load_metadata()
        # self.data_dir = '/work/guanqi/Genesis/scene1_2.3_3' 
        self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_0_9000_2.11.json'))
        self.data_dir = '/work/guanqi/Genesis/mpm_bounce_ball_videos6'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4') and int(f.split('_')[-6]) < 9000 and int(f.split('_')[-6]) in self.split_file['train']]
        self.metadata.sort()
        # ipdb.set_trace()
        spatial_transform = "resize"
        self.gt_ann_file = json.load(open('/work/guanqi/Genesis/mpm_bounce_ball_videos6/gt_retitution.json'))
        
        
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
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

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # sample = self.metadata.iloc[index]
            sample = self.metadata[index]
            # print(sample)
            # video_path = self._get_video_path(sample)
            video_path = os.path.join(self.data_dir, sample)
            ## video_path should be in the format of "....../WebVid/videos/$page_dir/$videoid.mp4"
            # caption = sample['caption']
            # caption = open_clip.tokenize('')
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
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            # start_idx = random.randint(0, random_range) if random_range > 0 else 0
            start_idx = 0
            frame_stride = 6

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # ipdb.set_trace()

        masks = (masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)

        # ipdb.set_trace()
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max


        gt_ann = self.gt_ann_file[video_path.split('_')[-6]]

        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask':masks, 'gt_ann':gt_ann}

        # ipdb.set_trace()
        # print(data['caption'])

        return data
    
    def __len__(self):
        return len(self.metadata)



class WebVidTest0(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 ):
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
        # self._load_metadata()
        # self.data_dir = '/work/guanqi/Genesis/scene1_2.3_3' 
        self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_0_9000_2.11.json'))
        self.data_dir = '/work/guanqi/Genesis/mpm_bounce_ball_videos6'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4') and int(f.split('_')[-6]) < 9000 and int(f.split('_')[-6]) in self.split_file['test']]
        self.metadata.sort()
        # ipdb.set_trace()
        spatial_transform = "resize"
        self.gt_ann_file = json.load(open('/work/guanqi/Genesis/mpm_bounce_ball_videos6/gt_retitution.json'))

        # ipdb.set_trace()
        
        
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
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

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # sample = self.metadata.iloc[index]
            sample = self.metadata[index]
            # print(sample)
            # video_path = self._get_video_path(sample)
            video_path = os.path.join(self.data_dir, sample)
            ## video_path should be in the format of "....../WebVid/videos/$page_dir/$videoid.mp4"
            # caption = sample['caption']
            # caption = open_clip.tokenize('')
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
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            # start_idx = random.randint(0, random_range) if random_range > 0 else 0
            start_idx = 0
            frame_stride = 6

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # ipdb.set_trace()

        masks = (masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)

        # ipdb.set_trace()
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max


        gt_ann = self.gt_ann_file[video_path.split('_')[-6]]

        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask':masks, 'gt_ann':gt_ann}

        # ipdb.set_trace()
        # print(data['caption'])

        return data
    
    def __len__(self):
        return len(self.metadata)


class WebVidTest1(WebVidTest0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override specific attributes
        self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_9000_10000_2.11.json'))
        self.data_dir = '/work/guanqi/Genesis/mpm_bounce_ball_videos6'
        
        # Reload metadata based on new split file
        self.metadata = [
            f for f in os.listdir(self.data_dir) 
            if f.endswith('rgb.mp4') and int(f.split('_')[-6]) in self.split_file['test']
        ]
        self.metadata.sort()

        self.gt_ann_file = json.load(open('/work/guanqi/Genesis/mpm_bounce_ball_videos6/gt_retitution.json'))



class WebVidTest2(WebVidTest0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override specific attributes
        self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_9000_10000_2.11.json'))
        self.data_dir = '/work/guanqi/Genesis/mpm_bounce_ball_videos6_test2'
        
        # Reload metadata based on new split file
        self.metadata = [
            f for f in os.listdir(self.data_dir) 
            if f.endswith('rgb.mp4') and int(f.split('_')[-6]) in self.split_file['test']
        ]
        self.metadata.sort()

        self.gt_ann_file = json.load(open('/work/guanqi/Genesis/mpm_bounce_ball_videos6_test2/gt_retitution.json'))



class WebVidTest3(WebVidTest0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override specific attributes
        self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_9000_10000_2.11.json'))
        self.data_dir = '/work/guanqi/Genesis/mpm_bounce_ball_videos6_test3'
        
        # Reload metadata based on new split file
        self.metadata = [
            f for f in os.listdir(self.data_dir) 
            if f.endswith('rgb.mp4') and int(f.split('_')[-6]) in self.split_file['test']
        ]
        self.metadata.sort()

        self.gt_ann_file = json.load(open('/work/guanqi/Genesis/mpm_bounce_ball_videos6_test3/gt_retitution.json'))


class WebVidTest4(WebVidTest0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override specific attributes
        self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_9000_10000_2.11.json'))
        self.data_dir = '/work/guanqi/Genesis/mpm_bounce_ball_videos6_test4'
        
        # Reload metadata based on new split file
        self.metadata = [
            f for f in os.listdir(self.data_dir) 
            if f.endswith('rgb.mp4') and int(f.split('_')[-6]) in self.split_file['test']
        ]
        self.metadata.sort()

        self.gt_ann_file = json.load(open('/work/guanqi/Genesis/mpm_bounce_ball_videos6_test4/gt_retitution.json'))




class WebVid217v1(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 ):
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
        # self._load_metadata()
        # self.data_dir = '/work/guanqi/Genesis/scene1_2.3_3' 
        # self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_0_9000_2.11.json'))
        # self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_4'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        self.metadata.sort()
        # ipdb.set_trace()
        spatial_transform = "resize"
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution.json')))
        
        
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
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

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # sample = self.metadata.iloc[index]
            sample = self.metadata[index]
            # print(sample)
            # video_path = self._get_video_path(sample)
            video_path = os.path.join(self.data_dir, sample)
            ## video_path should be in the format of "....../WebVid/videos/$page_dir/$videoid.mp4"
            # caption = sample['caption']
            # caption = open_clip.tokenize('')
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
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            # start_idx = random.randint(0, random_range) if random_range > 0 else 0
            start_idx = 0
            # frame_stride = 6 # for 100 frames video - elasticity
            # frame_stride = 9 # for 150 frames video - viscosity
            frame_stride = self.frame_stride


            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # ipdb.set_trace()

        masks = (masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)

        # ipdb.set_trace()
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max


        gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]
        # print(gt_ann)

        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask':masks, 'gt_ann':gt_ann}

        # ipdb.set_trace()
        # print(data['caption'])

        return data
    
    def __len__(self):
        return len(self.metadata)



# real benchmark
class WebVid217v1Test3(WebVid217v1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override specific attributes
        self.data_dir = '/work/guanqi/Genesis/real_elastic_benchmark_2.20'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        self.metadata.sort()

        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution.json')))

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # sample = self.metadata.iloc[index]
            sample = self.metadata[index]
            # print(sample)
            # video_path = self._get_video_path(sample)
            video_path = os.path.join(self.data_dir, sample)
            ## video_path should be in the format of "....../WebVid/videos/$page_dir/$videoid.mp4"
            # caption = sample['caption']
            # caption = open_clip.tokenize('')
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
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)

            if frame_num < self.video_length:
                print(f"Get frames failed! path = {video_path}; ")
                index += 1
                continue

            start_idx = 0
            frame_stride = frame_num // self.video_length

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # ipdb.set_trace()

        masks = (masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)

        # ipdb.set_trace()
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max


        gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]

        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask':masks, 'gt_ann':gt_ann}

        # ipdb.set_trace()
        # print(data['caption'])

        return data
    

class WebVid217v1Test1(WebVid217v1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override specific attributes
        self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_3_test1'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        self.metadata.sort()

        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution.json')))


class WebVid217v1Test2(WebVid217v1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override specific attributes
        self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_3_test2'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        self.metadata.sort()

        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution.json')))


class WebVid217v1Enhanced225(WebVid217v1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override specific attributes
        self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_4_incorporated_enhanced_2'
        self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        self.metadata.sort()

        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution.json')))




# compared with 217v1, use gt_retitution2.json instead, and also filter the meta data by the keys in the .json
class WebVid408v1(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 ):
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
        # self._load_metadata()
        # self.data_dir = '/work/guanqi/Genesis/scene1_2.3_3' 
        # self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_0_9000_2.11.json'))
        # self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_4'
        # self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        # self.metadata.sort()
        # ipdb.set_trace()
        spatial_transform = "resize"
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution2.json')))
        self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys()]
        self.metadata.sort()

        
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
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

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # sample = self.metadata.iloc[index]
            sample = self.metadata[index]
            # print(sample)
            # video_path = self._get_video_path(sample)
            video_path = os.path.join(self.data_dir, sample)
            ## video_path should be in the format of "....../WebVid/videos/$page_dir/$videoid.mp4"
            # caption = sample['caption']
            # caption = open_clip.tokenize('')
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
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            # start_idx = random.randint(0, random_range) if random_range > 0 else 0
            start_idx = 0
            # frame_stride = 6 # for 100 frames video - elasticity
            # frame_stride = 9 # for 150 frames video - viscosity
            frame_stride = self.frame_stride


            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # ipdb.set_trace()

        masks = (masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)

        # ipdb.set_trace()
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max


        gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]
        # print(gt_ann)

        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask':masks, 'gt_ann':gt_ann}

        # ipdb.set_trace()
        # print(data['caption'])

        return data
    
    def __len__(self):
        return len(self.metadata)




class WebVid408v1debug(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 ):
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
        # self._load_metadata()
        # self.data_dir = '/work/guanqi/Genesis/scene1_2.3_3' 
        # self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_0_9000_2.11.json'))
        # self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_4'
        # self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        # self.metadata.sort()
        # ipdb.set_trace()
        spatial_transform = "resize"
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution2.json')))
        self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys() if float(f.split('_')[3]) < 0.03]
        self.metadata.sort()

        
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
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

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # sample = self.metadata.iloc[index]
            sample = self.metadata[index]
            # print(sample)
            # video_path = self._get_video_path(sample)
            video_path = os.path.join(self.data_dir, sample)
            ## video_path should be in the format of "....../WebVid/videos/$page_dir/$videoid.mp4"
            # caption = sample['caption']
            # caption = open_clip.tokenize('')
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
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            # start_idx = random.randint(0, random_range) if random_range > 0 else 0
            start_idx = 0
            # frame_stride = 6 # for 100 frames video - elasticity
            # frame_stride = 9 # for 150 frames video - viscosity
            frame_stride = self.frame_stride


            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # ipdb.set_trace()

        masks = (masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)

        # ipdb.set_trace()
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max


        gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]
        # print(gt_ann)

        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask':masks, 'gt_ann':gt_ann}

        # ipdb.set_trace()
        # print(data['caption'])

        return data
    
    def __len__(self):
        return len(self.metadata)




class WebVidv2Elasticity(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 ):
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
        # self._load_metadata()
        # self.data_dir = '/work/guanqi/Genesis/scene1_2.3_3' 
        # self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_0_9000_2.11.json'))
        # self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_4'
        # self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        # self.metadata.sort()
        # ipdb.set_trace()
        spatial_transform = "resize"
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'bounce_analysis_results.json')))

        self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys()]
        self.metadata.sort()
        
        
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
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

        

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # sample = self.metadata.iloc[index]
            sample = self.metadata[index]
            # print(sample)
            # video_path = self._get_video_path(sample)
            video_path = os.path.join(self.data_dir, sample)
            ## video_path should be in the format of "....../WebVid/videos/$page_dir/$videoid.mp4"
            # caption = sample['caption']
            # caption = open_clip.tokenize('')
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
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            # start_idx = random.randint(0, random_range) if random_range > 0 else 0
            start_idx = 0
            # frame_stride = 6 # for 100 frames video - elasticity
            # frame_stride = 9 # for 150 frames video - viscosity
            frame_stride = self.frame_stride

            gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            # ipdb.set_trace()
            key_timestep_indices = gt_ann['temporal_key_points']
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                keytime_masks = mask_reader.get_batch(key_timestep_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        keytime_masks = torch.tensor(keytime_masks.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # ipdb.set_trace()

        masks = (masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        keytime_masks = (keytime_masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
            keytime_masks = self.spatial_transform(keytime_masks)

        # ipdb.set_trace()
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max


        
        # print(gt_ann)

        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask':masks, 'keytime_mask':keytime_masks, 'gt_ann':gt_ann}

        # ipdb.set_trace()
        # print(data['caption'])

        return data
    
    def __len__(self):
        return len(self.metadata)




class WebVidv2Viscosity(Dataset):
    """
    WebVid v2 单视频粘度评测数据集。

    无 red-circle / red2 / vjepa-red 时：磁盘上只需 *_rgb.mp4，mask 在 __getitem__ 内用全前景占位；
    否则仍要求 bounce 键对应 *_rgb.mp4 与 *_seg.mp4 同时存在（与旧逻辑一致）。
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 red_circle=False,
                 red_circle2=False, # one circle containing all frames
                 red_circle2_vjepa=False, # 使用VJEPA归一化方式
                 meta_name=None,
                 video_list_json=None,
                 ):
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
        # self._load_metadata()
        # self.data_dir = '/work/guanqi/Genesis/scene1_2.3_3' 
        # self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_0_9000_2.11.json'))
        # self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_4'
        # self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        # self.metadata.sort()
        # ipdb.set_trace()
        spatial_transform = "resize"
        if self.meta_name is None:
            self.meta_name = 'bounce_analysis_results.json'
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, self.meta_name)))

        self._no_red_eval = (
            not self.red_circle
            and not self.red_circle2
            and not self.red_circle2_vjepa
        )
        if self._no_red_eval:
            self.metadata = [
                f + "_rgb.mp4"
                for f in self.gt_ann_file.keys()
                if os.path.exists(os.path.join(self.data_dir, f + "_rgb.mp4"))
            ]
        else:
            self.metadata = [
                f + "_rgb.mp4"
                for f in self.gt_ann_file.keys()
                if os.path.exists(os.path.join(self.data_dir, f + "_rgb.mp4"))
                and os.path.exists(os.path.join(self.data_dir, f + "_seg.mp4"))
            ]
        self.metadata.sort()

        # Restrict evaluation to a fixed sample list when provided.
        if self.video_list_json is not None:
            try:
                selected_entries = json.load(open(self.video_list_json, "r"))
                selected_video_names = []
                for entry in selected_entries:
                    if isinstance(entry, str):
                        selected_video_names.append(entry + "_rgb.mp4")
                    elif isinstance(entry, (list, tuple)) and len(entry) > 0:
                        selected_video_names.append(str(entry[0]) + "_rgb.mp4")
                selected_set = set(selected_video_names)
                self.metadata = [name for name in self.metadata if name in selected_set]
            except (OSError, ValueError, TypeError):
                print(f"WARNING: failed to load video_list_json={self.video_list_json}, fallback to full metadata")
        print(len(self.metadata))

        # ipdb.set_trace()
        
        
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
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
        """
        按 index 取一条样本：RGB 视频张量、与帧同形的 mask（无 red 模式为全前景占位）、gt_ann 等。

        输入: index — 样本下标。
        输出: dict，含 video, mask, path, fps, gt_ann 等键。
        """
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # sample = self.metadata.iloc[index]
            sample = self.metadata[index]
            # print(sample)
            # video_path = self._get_video_path(sample)
            video_path = os.path.join(self.data_dir, sample)
            ## video_path should be in the format of "....../WebVid/videos/$page_dir/$videoid.mp4"
            # caption = sample['caption']
            # caption = open_clip.tokenize('')
            caption = ''
            if self.red_circle == False and self.red_circle2 == False:
                mask_path = video_path.replace('_rgb.mp4', '_seg.mp4')
            else:
                if self.red_circle:
                    mask_path = video_path.replace('_rgb.mp4', '_red.mp4')
                else:
                    assert self.red_circle2
                    mask_path = video_path.replace('_rgb.mp4', '_red2.mp4')

            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(
                        video_path, ctx=cpu(0), width=530, height=300
                    )
                mask_reader = None
                if not self._no_red_eval:
                    if self.load_raw_resolution:
                        mask_reader = VideoReader(mask_path, ctx=cpu(0))
                    else:
                        mask_reader = VideoReader(
                            mask_path, ctx=cpu(0), width=530, height=300
                        )
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except Exception:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            # start_idx = random.randint(0, random_range) if random_range > 0 else 0
            start_idx = 0
            # frame_stride = 6 # for 100 frames video - elasticity
            # frame_stride = 9 # for 150 frames video - viscosity
            frame_stride = self.frame_stride

            gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            # ipdb.set_trace()
            # key_timestep_indices = gt_ann['temporal_key_points']
            try:
                frames = video_reader.get_batch(frame_indices)
                if not self._no_red_eval:
                    masks = mask_reader.get_batch(frame_indices)
                break
            except Exception:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        fr_np = frames.asnumpy()
        frames = torch.tensor(fr_np).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        if self._no_red_eval:
            mk_np = np.full(fr_np.shape, 255, dtype=fr_np.dtype)
        else:
            mk_np = masks.asnumpy()
        masks = torch.tensor(mk_np).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        # keytime_masks = torch.tensor(keytime_masks.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # ipdb.set_trace()

        if self.red_circle == False and self.red_circle2 == False and self.red_circle2_vjepa == False:
            masks = (masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        # keytime_masks = (keytime_masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
            # keytime_masks = self.spatial_transform(keytime_masks)

        # ipdb.set_trace()
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2

        # if self.red_circle or self.red_circle2:
        #     masks = (masks / 255 - 0.5) * 2
        # elif self.red_circle2_vjepa:
        if self.red_circle2_vjepa:
            # 应用VJEPA归一化方式
            masks = masks / 255.0  # [0,255] -> [0,1]
            # 对每个通道应用ImageNet标准化 (此时masks维度为[C,T,H,W]，第0维是通道维度)
            for c in range(min(masks.shape[0], 3)):  # 只处理前3个通道(RGB)
                masks[c] = (masks[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]

        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max


        
        # print(gt_ann)

        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask':masks, 'keytime_mask':masks, 'gt_ann':gt_ann}

        # ipdb.set_trace()
        # print(data['caption'])

        return data
    
    def __len__(self):
        return len(self.metadata)



class WebVidv3Viscosity(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 red_circle=False,
                 red_circle2=False,
                 red_circle2_vjepa=False, # 使用VJEPA归一化方式
                 video_list_json=None,
                 ):
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
        # self._load_metadata()
        # self.data_dir = '/work/guanqi/Genesis/scene1_2.3_3' 
        # self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_0_9000_2.11.json'))
        # self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_4'
        # self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        # self.metadata.sort()
        # ipdb.set_trace()
        # spatial_transform = "resize"
        # self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'bounce_analysis_results.json')))

        # self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys() if (os.path.exists(self.data_dir, f + '_rgb.mp4') and os.path.exists(self.data_dir, f + '_seg.mp4'))]
        
        # TODO COMPLETED: Create mapping from video indices to basenames for efficient sampling
        self.video_groups = {}  # Maps video_i to list of available video_j basenames
        self.all_basenames = []
        
        # Read basenames directly from directory listing
        all_files = os.listdir(self.data_dir)
        rgb_files = [f for f in all_files if f.endswith('_rgb.mp4')]
        # 与 __getitem__ 一致：seg / red / red2 三种掩膜文件（5.24 friction synthetic 等为 _red2）
        if self.red_circle2 or self.red_circle2_vjepa:
            mask_pair_name = "_red2.mp4"
        elif self.red_circle:
            mask_pair_name = "_red.mp4"
        else:
            mask_pair_name = "_seg.mp4"
        
        for rgb_file in rgb_files:
            basename = rgb_file[:-8]  # Remove '_rgb.mp4' suffix
            mask_file = basename + mask_pair_name
            
            # Check if both rgb and mask companion exist
            if mask_file in all_files:
                
                # Extract video_i from basename (format: video_i_id_j_...)
                parts = basename.split('_')
                if len(parts) >= 3 and parts[0] == 'video':
                    try:
                        video_i = int(parts[1])
                        if video_i not in self.video_groups:
                            self.video_groups[video_i] = []
                        self.video_groups[video_i].append(basename)
                        self.all_basenames.append(basename)
                        # if len(self.video_groups[video_i]) > 5:
                        #     ipdb.set_trace()
                    except ValueError:
                        # Skip if video_i is not a valid integer
                        continue
            # else:
            #     ipdb.set_trace()
        
        # Filter to only keep video_i groups that have at least 2 videos for comparison
        self.valid_video_groups = {k: v for k, v in self.video_groups.items() if len(v) >= 2}
        
        # ipdb.set_trace()
        # Create metadata list based on valid video groups
        self.metadata = list(self.valid_video_groups.keys())
        print(f"Found {len(self.metadata)} video groups with multiple videos for comparison")

        # Restrict training to explicit basename pairs if a JSON list is provided (aligned with Test).
        if self.video_list_json is not None:
            try:
                selected_entries = json.load(open(self.video_list_json, "r"))
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
                    print(
                        f"Using explicit pair list from {self.video_list_json}, "
                        f"total pairs: {len(self.metadata)}"
                    )
                else:
                    print(
                        f"WARNING: video_list_json={self.video_list_json} matched 0 pairs "
                        "in data_dir; keeping grouped video_i metadata (train sampler)."
                    )
            except (OSError, ValueError, TypeError):
                print(
                    f"WARNING: failed to load video_list_json={self.video_list_json}, "
                    "fallback to grouped metadata"
                )
        
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Resize(self.resolution)
            elif spatial_transform == "extreme_augmentation":
                # self.spatial_transform1 = transforms.RandomResizedCrop(
                #         size=self.resolution,
                #         scale=(0.8, 1.0),  # Very aggressive cropping
                #         ratio=(1.4, 1.6),  # Wide range of aspect ratios
                #     )
                # self.spatial_transform2 = transforms.RandomHorizontalFlip(p=0.5),
                # self.spatial_transform3 = transforms.RandomVerticalFlip(p=0.3)
                # self.spatial_transform4 = transforms.RandomRotation(degrees=30)
                # self.spatial_transform5 = transforms.RandomPerspective(distortion_scale=0.3, p=0.5)
                # self.spatial_transform6 = transforms.ColorJitter(
                #         brightness=0.4,
                #         contrast=0.4,
                #         saturation=0.4,
                #         hue=0.2
                #     )
                # self.spatial_transform7 = transforms.RandomGrayscale(p=0.1)
                # self.spatial_transform8 = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                self.spatial_transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        size=self.resolution,
                        scale=(0.8, 1.0),  # Very aggressive cropping
                        ratio=(1.4, 1.6),  # Wide range of aspect ratios
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(degrees=30),
                    # transforms.RandomAffine(
                    #     degrees=15,
                    #     translate=(0.01, 0.01),
                    #     # scale=(0.8, 1.2),
                    #     shear=10
                    # ),
                    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.2
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                ])
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
        """Extract property value from basename (assumes it's the 8th element when split by '_')"""
        try:
            return float(basename.split('_')[7])
        except (IndexError, ValueError):
            return 0.0
    
    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # index = 1

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

                # Randomly sample 2 different videos from this group
                if len(available_basenames) < 2:
                    index += 1
                    continue

                sampled_basenames = random.sample(available_basenames, 2)
                video_basename1 = sampled_basenames[0]
                video_basename2 = sampled_basenames[1]

            # Get property values for comparison
            prop_value1 = self._get_property_value(video_basename1)
            prop_value2 = self._get_property_value(video_basename2)

            # Check if ratio constraint is satisfied (larger/smaller < 1.2 means re-sample)
            if (not explicit_pair_mode) and prop_value1 != 0 and prop_value2 != 0:
                ratio = max(prop_value1, prop_value2) / min(prop_value1, prop_value2)
                if ratio < 1.2:
                    index += 1
                    continue

            video_path1 = os.path.join(self.data_dir, video_basename1 + '_rgb.mp4')
            video_path2 = os.path.join(self.data_dir, video_basename2 + '_rgb.mp4')
            if self.red_circle == False and self.red_circle2 == False:
                mask_path1 = os.path.join(self.data_dir, video_basename1 + '_seg.mp4')
                mask_path2 = os.path.join(self.data_dir, video_basename2 + '_seg.mp4')
            else:
                if self.red_circle:
                    mask_path1 = os.path.join(self.data_dir, video_basename1 + '_red.mp4')
                    mask_path2 = os.path.join(self.data_dir, video_basename2 + '_red.mp4')
                else:
                    assert self.red_circle2
                    mask_path1 = os.path.join(self.data_dir, video_basename1 + '_red2.mp4')
                    mask_path2 = os.path.join(self.data_dir, video_basename2 + '_red2.mp4')

            # TODO COMPLETED: Process both videos and stack them
            caption = ''
            
            try:
                # Load both videos and masks
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
                
                # Check if both videos have sufficient length
                if len(video_reader1) < self.video_length or len(video_reader2) < self.video_length:
                    print(f"Video length too small: {len(video_reader1)}, {len(video_reader2)} < {self.video_length}")
                    index += 1
                    continue
                    
            except Exception as e:
                index += 1
                print(f"Load video failed! paths = {video_path1}, {video_path2}, error: {e}")
                continue
            
            
            fps_ori = video_reader1.get_avg_fps()
            start_idx = 0
            frame_stride = self.frame_stride

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            
            try:
                frames1 = video_reader1.get_batch(frame_indices)
                masks1 = mask_reader1.get_batch(frame_indices)
                frames2 = video_reader2.get_batch(frame_indices)
                masks2 = mask_reader2.get_batch(frame_indices)
                break
            except Exception as e:
                print(f"Get frames failed! paths = {video_path1}, {video_path2}, error: {e}")
                index += 1
                continue
        
        ## process data for both videos
        assert(frames1.shape[0] == self.video_length and frames2.shape[0] == self.video_length), \
            f'Frame lengths: {frames1.shape[0]}, {frames2.shape[0]}, expected: {self.video_length}'
        
        # Convert to tensors and permute dimensions
        frames1 = torch.tensor(frames1.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks1 = torch.tensor(masks1.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        frames2 = torch.tensor(frames2.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks2 = torch.tensor(masks2.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # Process masks
        if self.red_circle == False and self.red_circle2 == False and self.red_circle2_vjepa == False:
            masks1 = (masks1.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
            masks2 = (masks2.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            # ipdb.set_trace()
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
            assert (frames1.shape[2], frames1.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames1={frames1.shape}, self.resolution={self.resolution}'
            assert (frames2.shape[2], frames2.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames2={frames2.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames1 = (frames1 / 255 - 0.5) * 2
        frames2 = (frames2 / 255 - 0.5) * 2

        if self.red_circle or self.red_circle2:
            masks1 = (masks1 / 255 - 0.5) * 2
            masks2 = (masks2 / 255 - 0.5) * 2
        elif self.red_circle2_vjepa:
            # 应用VJEPA归一化方式
            masks1 = masks1 / 255.0  # [0,255] -> [0,1]
            masks2 = masks2 / 255.0  # [0,255] -> [0,1]
            # 对每个通道应用ImageNet标准化 (此时masks维度为[C,T,H,W]，第0维是通道维度)
            for c in range(min(masks1.shape[0], 3)):  # 只处理前3个通道(RGB)
                masks1[c] = (masks1[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]
                masks2[c] = (masks2[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]
        
        # Stack the two videos along a new dimension (first dimension after batch)
        frames = torch.stack([frames1, frames2], dim=0)  # [2, c, t, h, w]
        masks = torch.stack([masks1, masks2], dim=0)     # [2, 1, t, h, w]
        
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        # TODO COMPLETED: Create ground truth annotation for comparison
        # gt_restitution: 1 if property value of video1 > property value of video2, 0 otherwise
        gt_restitution = 1 if prop_value1 > prop_value2 else 0
        gt_ann = {'gt_restitution': gt_restitution, 
                  'prop_value1': prop_value1, 
                  'prop_value2': prop_value2,
                  'video_basename1': video_basename1,
                  'video_basename2': video_basename2}

        # print(gt_ann)

        data = {'video': frames, 
                'caption': caption, 
                'path': video_path1,  # Keep path from first video for reference
                'fps': fps_clip, 
                'frame_stride': frame_stride, 
                'mask': masks, 
                'keytime_mask': masks, 
                'gt_ann': gt_ann}

        return data
    
    def __len__(self):
        return len(self.metadata)
    


class WebVidv3ViscosityTest(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 red_circle=False,
                 red_circle2=False,
                 red_circle2_vjepa=False, # 使用VJEPA归一化方式
                 video_list_json=None,
                 ):
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
        # self._load_metadata()
        # self.data_dir = '/work/guanqi/Genesis/scene1_2.3_3' 
        # self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_0_9000_2.11.json'))
        # self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_4'
        # self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        # self.metadata.sort()
        # ipdb.set_trace()
        spatial_transform = "resize"
        # self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'bounce_analysis_results.json')))

        # self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys() if (os.path.exists(self.data_dir, f + '_rgb.mp4') and os.path.exists(self.data_dir, f + '_seg.mp4'))]
        
        # TODO COMPLETED: Create mapping from video indices to basenames for efficient sampling
        self.video_groups = {}  # Maps video_i to list of available video_j basenames
        self.all_basenames = []
        
        # Read basenames directly from directory listing
        all_files = os.listdir(self.data_dir)
        all_files.sort()
        rgb_files = [f for f in all_files if f.endswith('_rgb.mp4')]
        if self.red_circle2 or self.red_circle2_vjepa:
            mask_pair_name = "_red2.mp4"
        elif self.red_circle:
            mask_pair_name = "_red.mp4"
        else:
            mask_pair_name = "_seg.mp4"
        
        for rgb_file in rgb_files:
            basename = rgb_file[:-8]  # Remove '_rgb.mp4' suffix
            mask_file = basename + mask_pair_name
            
            if mask_file in all_files:
                
                # Extract video_i from basename (format: video_i_id_j_...)
                parts = basename.split('_')
                if len(parts) >= 3 and parts[0] == 'video':
                    try:
                        video_i = int(parts[1])
                        if video_i not in self.video_groups:
                            self.video_groups[video_i] = []
                        self.video_groups[video_i].append(basename)
                        self.all_basenames.append(basename)
                    except ValueError:
                        # Skip if video_i is not a valid integer
                        continue
        
        # Filter to only keep video_i groups that have at least 2 videos for comparison
        self.valid_video_groups = {k: v for k, v in self.video_groups.items() if len(v) >= 2}
        
        # Create metadata list based on valid video groups
        self.metadata = list(self.valid_video_groups.keys())
        print(f"Found {len(self.metadata)} video groups with multiple videos for comparison")

        # Restrict evaluation to explicit basename pairs if a JSON list is provided.
        if self.video_list_json is not None:
            try:
                selected_entries = json.load(open(self.video_list_json, "r"))
                selected_pairs = []
                all_basename_set = set(self.all_basenames)
                for entry in selected_entries:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        b1 = str(entry[0])
                        b2 = str(entry[1])
                        if b1 in all_basename_set and b2 in all_basename_set:
                            selected_pairs.append([b1, b2])
                self.metadata = selected_pairs
                print(f"Using explicit pair list from {self.video_list_json}, total pairs: {len(self.metadata)}")
            except (OSError, ValueError, TypeError):
                print(f"WARNING: failed to load video_list_json={self.video_list_json}, fallback to grouped metadata")
        
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
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
        """Extract property value from basename (assumes it's the 8th element when split by '_')"""
        try:
            return float(basename.split('_')[7])
        except (IndexError, ValueError):
            return 0.0
    
    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # index = 1
            
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
                
                # Randomly sample 2 different videos from this group
                if len(available_basenames) < 2:
                    index += 1
                    continue
                
                # sampled_basenames = random.sample(available_basenames, 2) # TODO: instead of randomly sample 2 videos, emunulate every possible combinations (order reversed counts as 2, so should be 5 x 4 = 20 combinations here), if for one combination it is not good, continue for the next combination, still for the same index
                # always use No.0 and No.1 for an initial test
                video_basename1 = available_basenames[0]
                video_basename2 = available_basenames[1]
            
            # Get property values for comparison
            prop_value1 = self._get_property_value(video_basename1)
            prop_value2 = self._get_property_value(video_basename2)
            
            # Check if ratio constraint is satisfied (larger/smaller < 1.2 means re-sample)
            if (not explicit_pair_mode) and prop_value1 != 0 and prop_value2 != 0:
                ratio = max(prop_value1, prop_value2) / min(prop_value1, prop_value2)
                if ratio < 1.2:
                    index += 1
                    continue

            video_path1 = os.path.join(self.data_dir, video_basename1 + '_rgb.mp4')
            video_path2 = os.path.join(self.data_dir, video_basename2 + '_rgb.mp4')
            if self.red_circle == False and self.red_circle2 == False:
                mask_path1 = os.path.join(self.data_dir, video_basename1 + '_seg.mp4')
                mask_path2 = os.path.join(self.data_dir, video_basename2 + '_seg.mp4')
            else:
                if self.red_circle:
                    mask_path1 = os.path.join(self.data_dir, video_basename1 + '_red.mp4')
                    mask_path2 = os.path.join(self.data_dir, video_basename2 + '_red.mp4')
                else:
                    assert self.red_circle2
                    mask_path1 = os.path.join(self.data_dir, video_basename1 + '_red2.mp4')
                    mask_path2 = os.path.join(self.data_dir, video_basename2 + '_red2.mp4')

            # TODO COMPLETED: Process both videos and stack them
            caption = ''
            
            try:
                # Load both videos and masks
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
                
                # Check if both videos have sufficient length
                if len(video_reader1) < self.video_length or len(video_reader2) < self.video_length:
                    print(f"Video length too small: {len(video_reader1)}, {len(video_reader2)} < {self.video_length}")
                    index += 1
                    continue
                    
            except Exception as e:
                index += 1
                print(f"Load video failed! paths = {video_path1}, {video_path2}, error: {e}")
                continue
            
            
            fps_ori = video_reader1.get_avg_fps()
            start_idx = 0
            frame_stride = self.frame_stride

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            
            try:
                frames1 = video_reader1.get_batch(frame_indices)
                masks1 = mask_reader1.get_batch(frame_indices)
                frames2 = video_reader2.get_batch(frame_indices)
                masks2 = mask_reader2.get_batch(frame_indices)
                break
            except Exception as e:
                print(f"Get frames failed! paths = {video_path1}, {video_path2}, error: {e}")
                index += 1
                continue
        
        ## process data for both videos
        assert(frames1.shape[0] == self.video_length and frames2.shape[0] == self.video_length), \
            f'Frame lengths: {frames1.shape[0]}, {frames2.shape[0]}, expected: {self.video_length}'
        
        # Convert to tensors and permute dimensions
        frames1 = torch.tensor(frames1.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks1 = torch.tensor(masks1.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        frames2 = torch.tensor(frames2.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks2 = torch.tensor(masks2.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # Process masks
        if self.red_circle == False and self.red_circle2 == False and self.red_circle2_vjepa == False:
            masks1 = (masks1.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
            masks2 = (masks2.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            frames1 = self.spatial_transform(frames1)
            masks1 = self.spatial_transform(masks1)
            frames2 = self.spatial_transform(frames2)
            masks2 = self.spatial_transform(masks2)

        if self.resolution is not None:
            assert (frames1.shape[2], frames1.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames1={frames1.shape}, self.resolution={self.resolution}'
            assert (frames2.shape[2], frames2.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames2={frames2.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames1 = (frames1 / 255 - 0.5) * 2
        frames2 = (frames2 / 255 - 0.5) * 2

        # if self.red_circle or self.red_circle2:
        #     masks1 = (masks1 / 255 - 0.5) * 2
        #     masks2 = (masks2 / 255 - 0.5) * 2
        # elif self.red_circle2_vjepa:
        if self.red_circle2_vjepa:
            # 应用VJEPA归一化方式
            masks1 = masks1 / 255.0  # [0,255] -> [0,1]
            masks2 = masks2 / 255.0  # [0,255] -> [0,1]
            # 对每个通道应用ImageNet标准化 (此时masks维度为[C,T,H,W]，第0维是通道维度)
            for c in range(min(masks1.shape[0], 3)):  # 只处理前3个通道(RGB)
                masks1[c] = (masks1[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]
                masks2[c] = (masks2[c] - IMAGENET_DEFAULT_MEAN[c]) / IMAGENET_DEFAULT_STD[c]
        
        # Stack the two videos along a new dimension (first dimension after batch)
        frames = torch.stack([frames1, frames2], dim=0)  # [2, c, t, h, w]
        masks = torch.stack([masks1, masks2], dim=0)     # [2, 1, t, h, w]
        
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        # TODO COMPLETED: Create ground truth annotation for comparison
        # gt_restitution: 1 if property value of video1 > property value of video2, 0 otherwise
        gt_restitution = 1 if prop_value1 > prop_value2 else 0
        gt_ann = {'gt_restitution': gt_restitution, 
                  'prop_value1': prop_value1, 
                  'prop_value2': prop_value2,
                  'video_basename1': video_basename1,
                  'video_basename2': video_basename2}

        # print(gt_ann)
        

        # import matplotlib.pyplot as plt

        # save_dir = '/work/guanqi/DynamiCrafter/vis_7.20_3'
        # os.makedirs(save_dir, exist_ok=True)

        # fig, axes = plt.subplots(4, 8, figsize=(24, 12))  # 4行8列显示16帧对比
        # axes = axes.flatten()

        # for t in range(16):
        #     # 上半部分显示masks1
        #     if t < 8:
        #         axes[t].imshow(masks1[0, t].cpu().numpy(), cmap='gray')
        #         axes[t].set_title(f'masks1-{t}', fontsize=8)
        #     else:
        #         # 下半部分显示masks2
        #         axes[t+8].imshow(masks2[0, t-8].cpu().numpy(), cmap='gray')
        #         axes[t+8].set_title(f'masks2-{t-8}', fontsize=8)
            
        #     axes[t].axis('off')
        #     if t < 8:
        #         axes[t+16].imshow(masks2[0, t].cpu().numpy(), cmap='gray')
        #         axes[t+16].set_title(f'masks2-{t}', fontsize=8)
        #         axes[t+16].axis('off')

        # plt.tight_layout()
        # plt.savefig(f"{save_dir}/all_frames_grid.png", dpi=150, bbox_inches='tight')
        # plt.close()

        # ipdb.set_trace()

        save_name_path = '_'.join(os.path.basename(video_path1).split('_')[:4]) + '--' + '_'.join(os.path.basename(video_path2).split('_')[:4])

        data = {'video': frames, 
                'caption': caption, 
                'path': save_name_path,  # Keep path from first video for reference
                'fps': fps_clip, 
                'frame_stride': frame_stride, 
                'mask': masks, 
                'keytime_mask': masks, 
                'gt_ann': gt_ann}

        return data
    
    def __len__(self):
        return len(self.metadata)
    

class WebVidv3Friction(WebVidv3Viscosity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _get_property_value(self, basename):
        """Extract property value from basename (assumes it's the 8th element when split by '_')"""
        try:
            return float(basename.split('_')[8])
        except (IndexError, ValueError):
            return 0.0

class WebVidv3Elasticity(WebVidv3Viscosity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_file = json.load(open(os.path.join(self.data_dir, 'gt_retitution1.json')))
    
    def _get_property_value(self, basename):
        """Extract property value from basename (assumes it's the 8th element when split by '_')"""
        return self.gt_file[basename]


class WebVidv3FrictionTest(WebVidv3ViscosityTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _get_property_value(self, basename):
        """Extract property value from basename (assumes it's the 8th element when split by '_')"""
        try:
            return float(basename.split('_')[8])
        except (IndexError, ValueError):
            return 0.0


class WebVidv3ElasticityTest(WebVidv3ViscosityTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_file = json.load(open(os.path.join(self.data_dir, 'gt_restitution1.json')))
    
    def _get_property_value(self, basename):
        """Extract property value from basename (assumes it's the 8th element when split by '_')"""
        return self.gt_file[basename]
    
    def __getitem__(self, index):
        """
        调用父类实现；当提供 video_list_json 时，父类会按显式 pair 逐项评估并跳过 ratio 过滤。
        """
        return super().__getitem__(index)



class WebVidv3VFRealTest(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 ):
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
        # self._load_metadata()
        # self.data_dir = '/work/guanqi/Genesis/scene1_2.3_3' 
        # self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_0_9000_2.11.json'))
        # self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_4'
        # self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        # self.metadata.sort()
        # ipdb.set_trace()
        spatial_transform = "resize"
        # self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'bounce_analysis_results.json')))

        # self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys() if (os.path.exists(self.data_dir, f + '_rgb.mp4') and os.path.exists(self.data_dir, f + '_seg.mp4'))]
        
        # TODO COMPLETED: Create mapping from video indices to basenames for efficient sampling
        self.video_groups = {}  # Maps video_i to list of available video_j basenames
        self.all_basenames = []
        
        # Read basenames directly from directory listing
        all_files = os.listdir(self.data_dir)
        all_files.sort()
        rgb_files = [f for f in all_files if f.endswith('.mp4')]
        
        for rgb_file in rgb_files:
            basename = rgb_file[:-4]  # Remove '_rgb.mp4' suffix
            
            # Extract video_i from basename (format: video_i_id_j_...)
            parts = basename.split('_')
            if len(parts) >= 3 and parts[0] == 'video':
                try:
                    video_i = int(parts[1])
                    if video_i not in self.video_groups:
                        self.video_groups[video_i] = []
                    self.video_groups[video_i].append(basename)
                    self.all_basenames.append(basename)
                except ValueError:
                    # Skip if video_i is not a valid integer
                    continue
        
        # Filter to only keep video_i groups that have at least 2 videos for comparison
        self.valid_video_groups = {k: v for k, v in self.video_groups.items() if len(v) >= 2}
        
        # Create metadata list based on valid video groups
        self.metadata = list(self.valid_video_groups.keys())
        print(f"Found {len(self.metadata)} video groups with multiple videos for comparison")
        
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
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
        """Extract property value from basename (assumes it's the 8th element when split by '_')"""
        try:
            return float(basename.split('_')[8])
        except (IndexError, ValueError):
            return 0.0
    
    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # index = 1
            
            video_i = self.metadata[index]
            available_basenames = self.valid_video_groups[video_i]
            
            # Randomly sample 2 different videos from this group
            if len(available_basenames) < 2:
                index += 1
                continue
                
            # sampled_basenames = random.sample(available_basenames, 2) # TODO: instead of randomly sample 2 videos, emunulate every possible combinations (order reversed counts as 2, so should be 5 x 4 = 20 combinations here), if for one combination it is not good, continue for the next combination, still for the same index
            # always use No.0 and No.1 for an initial test
            video_basename1 = available_basenames[0]
            video_basename2 = available_basenames[1]
            
            # Get property values for comparison
            prop_value1 = self._get_property_value(video_basename1)
            prop_value2 = self._get_property_value(video_basename2)
            
            # Check if ratio constraint is satisfied (larger/smaller < 1.2 means re-sample)
            if prop_value1 != 0 and prop_value2 != 0:
                ratio = max(prop_value1, prop_value2) / min(prop_value1, prop_value2)
                if ratio < 1.2:
                    index += 1
                    continue

            video_path1 = os.path.join(self.data_dir, video_basename1 + '.mp4')
            video_path2 = os.path.join(self.data_dir, video_basename2 + '.mp4')
        

            # TODO COMPLETED: Process both videos and stack them
            caption = ''
            
            try:
                # Load both videos and masks
                if self.load_raw_resolution:
                    video_reader1 = VideoReader(video_path1, ctx=cpu(0))
                    
                    video_reader2 = VideoReader(video_path2, ctx=cpu(0))
                    
                else:
                    video_reader1 = VideoReader(video_path1, ctx=cpu(0), width=530, height=300)
                    
                    video_reader2 = VideoReader(video_path2, ctx=cpu(0), width=530, height=300)
                    
                
                # Check if both videos have sufficient length
                if len(video_reader1) < self.video_length or len(video_reader2) < self.video_length:
                    print(f"Video length too small: {len(video_reader1)}, {len(video_reader2)} < {self.video_length}")
                    index += 1
                    continue
                    
            except Exception as e:
                index += 1
                print(f"Load video failed! paths = {video_path1}, {video_path2}, error: {e}")
                continue
            
            
            fps_ori = video_reader1.get_avg_fps()
            start_idx = 0
            frame_stride = self.frame_stride

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            
            try:
                frames1 = video_reader1.get_batch(frame_indices)
                
                frames2 = video_reader2.get_batch(frame_indices)
                
                break
            except Exception as e:
                print(f"Get frames failed! paths = {video_path1}, {video_path2}, error: {e}")
                index += 1
                continue
        
        ## process data for both videos
        assert(frames1.shape[0] == self.video_length and frames2.shape[0] == self.video_length), \
            f'Frame lengths: {frames1.shape[0]}, {frames2.shape[0]}, expected: {self.video_length}'
        
        # Convert to tensors and permute dimensions
        frames1 = torch.tensor(frames1.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        # masks1 = torch.tensor(masks1.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        frames2 = torch.tensor(frames2.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        # masks2 = torch.tensor(masks2.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # # Process masks
        # masks1 = (masks1.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        # masks2 = (masks2.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            frames1 = self.spatial_transform(frames1)
            # masks1 = self.spatial_transform(masks1)
            frames2 = self.spatial_transform(frames2)
            # masks2 = self.spatial_transform(masks2)

        if self.resolution is not None:
            assert (frames1.shape[2], frames1.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames1={frames1.shape}, self.resolution={self.resolution}'
            assert (frames2.shape[2], frames2.shape[3]) == (self.resolution[0], self.resolution[1]), \
                f'frames2={frames2.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames1 = (frames1 / 255 - 0.5) * 2
        frames2 = (frames2 / 255 - 0.5) * 2
        
        # Stack the two videos along a new dimension (first dimension after batch)
        frames = torch.stack([frames1, frames2], dim=0)  # [2, c, t, h, w]
        # masks = torch.stack([masks1, masks2], dim=0)     # [2, 1, t, h, w]
        
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        # TODO COMPLETED: Create ground truth annotation for comparison
        # gt_restitution: 1 if property value of video1 > property value of video2, 0 otherwise
        gt_restitution = 1 if prop_value1 > prop_value2 else 0
        gt_ann = {'gt_restitution': gt_restitution, 
                  'prop_value1': prop_value1, 
                  'prop_value2': prop_value2,
                  'video_basename1': video_basename1,
                  'video_basename2': video_basename2}

        # print(gt_ann)

        save_name_path = '_'.join(os.path.basename(video_path1).split('_')[:4]) + '--' + '_'.join(os.path.basename(video_path2).split('_')[:4])

        data = {'video': frames, 
                'caption': caption, 
                'path': save_name_path,  # Keep path from first video for reference
                'fps': fps_clip, 
                'frame_stride': frame_stride, 
                'mask': frames, 
                'keytime_mask': frames, 
                'gt_ann': gt_ann}

        return data
    
    def __len__(self):
        return len(self.metadata)
    


class WebVidv2Elasticitydebug(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 ):
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
        # self._load_metadata()
        # self.data_dir = '/work/guanqi/Genesis/scene1_2.3_3' 
        # self.split_file = json.load(open('/work/guanqi/Genesis/train_test_random_split_0_9000_2.11.json'))
        # self.data_dir = '/scratch/shared/beegfs/guanqi/genesis_data/2.16_mpm_bounce_ball_videos_4'
        # self.metadata = [f for f in os.listdir(self.data_dir) if f.endswith('rgb.mp4')]
        # self.metadata.sort()
        # ipdb.set_trace()
        spatial_transform = "resize"
        self.gt_ann_file = json.load(open(os.path.join(self.data_dir, 'bounce_analysis_results.json')))

        self.metadata = [f + '_rgb.mp4' for f in self.gt_ann_file.keys() if float(f.split('_')[3]) < 0.03]
        self.metadata.sort()
        
        
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
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

        

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            # sample = self.metadata.iloc[index]
            sample = self.metadata[index]
            # print(sample)
            # video_path = self._get_video_path(sample)
            video_path = os.path.join(self.data_dir, sample)
            ## video_path should be in the format of "....../WebVid/videos/$page_dir/$videoid.mp4"
            # caption = sample['caption']
            # caption = open_clip.tokenize('')
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
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            # start_idx = random.randint(0, random_range) if random_range > 0 else 0
            start_idx = 0
            # frame_stride = 6 # for 100 frames video - elasticity
            # frame_stride = 9 # for 150 frames video - viscosity
            frame_stride = self.frame_stride

            gt_ann = self.gt_ann_file[video_path.split('/')[-1][:-8]]

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            # ipdb.set_trace()
            key_timestep_indices = gt_ann['temporal_key_points']
            try:
                frames = video_reader.get_batch(frame_indices)
                masks = mask_reader.get_batch(frame_indices)
                keytime_masks = mask_reader.get_batch(key_timestep_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        masks = torch.tensor(masks.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        keytime_masks = torch.tensor(keytime_masks.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]

        # ipdb.set_trace()

        masks = (masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        keytime_masks = (keytime_masks.mean(dim=0).unsqueeze(0) > 128) # [1,t,h,w]
        
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
            masks = self.spatial_transform(masks)
            keytime_masks = self.spatial_transform(keytime_masks)

        # ipdb.set_trace()
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max


        
        # print(gt_ann)

        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'mask':masks, 'keytime_mask':keytime_masks, 'gt_ann':gt_ann}

        # ipdb.set_trace()
        # print(data['caption'])

        return data
    
    def __len__(self):
        return len(self.metadata)





if __name__== "__main__":
    meta_path = "" ## path to the meta file
    data_dir = "" ## path to the data directory
    save_dir = "" ## path to the save directory
    dataset = WebVid(meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256,448],
                 frame_stride=4,
                 spatial_transform="resize_center_crop",
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=True
                 )
    dataloader = DataLoader(dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=False)

    
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
    from utils.save_video import tensor_to_mp4
    for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
        video = batch['video']
        name = batch['path'][0].split('videos/')[-1].replace('/','_')
        tensor_to_mp4(video, save_dir+'/'+name, fps=8)

