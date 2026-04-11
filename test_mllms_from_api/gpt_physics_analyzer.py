import os
import random
import json
import re
import time
import cv2
import base64
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

class GPTPhysicsAnalyzer:

    def __init__(self, config_path: str='vjepa_data_dirs.json'):
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        self.config_path = config_path
        self.load_config()
        self.properties = ['friction', 'viscosity', 'elasticity']
        self.analysis_types = ['absolute', 'relative']
        self.dataset_gt_mapping = {'WebVidv2Viscosity': 'bounce_analysis_results.json', 'WebVidv3ViscosityTest': 'bounce_analysis_results.json', 'WebVidv3FrictionTest': 'bounce_analysis_results.json', 'WebVidv3ElasticityTest': 'gt_retitution1.json'}
        self.version_dataset_mapping = {'v119_vjepa': 'WebVidv2Viscosity', 'v110_vjepa': 'WebVidv3ViscosityTest', 'v120_vjepa': 'WebVidv2Viscosity', 'v121_vjepa': 'WebVidv3FrictionTest', 'v124_vjepa': 'WebVidv2Viscosity', 'v125_vjepa': 'WebVidv3ElasticityTest'}
        self.version_gt_override = {'v124_vjepa': 'gt_retitution1.json'}

    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        print(f'Loaded configuration:{len(self.config)}test version')

    def get_gt_data(self, data_dir: str, dataset_class: str, property_type: str, version: str=None) -> Optional[Dict]:
        if version and version in self.version_gt_override:
            gt_file = self.version_gt_override[version]
        else:
            gt_file = self.dataset_gt_mapping.get(dataset_class)
        if gt_file:
            gt_path = os.path.join(data_dir, gt_file)
            if os.path.exists(gt_path):
                with open(gt_path, 'r') as f:
                    return json.load(f)
            else:
                print(f'Warning: GT file does not exist -{gt_path}')
                return None
        else:
            print(f'Parse GT data using file names, dataset class:{dataset_class}')
            return {}

    def extract_property_from_filename(self, filename: str, property_type: str, dataset_class: str) -> float:
        basename = filename.replace('_rgb.mp4', '')
        parts = basename.split('_')
        try:
            if dataset_class in ['WebVidv3ViscosityTest', 'WebVidv3FrictionTest']:
                if 'objFriction' in parts:
                    obj_friction_index = parts.index('objFriction')
                    if obj_friction_index + 1 < len(parts):
                        return float(parts[obj_friction_index + 1])
                for i, part in enumerate(parts):
                    try:
                        val = float(part)
                        if val > 0 and val not in [0, 1, 2] and ('.' in part):
                            return val
                    except ValueError:
                        continue
                return 0.0
            elif dataset_class == 'WebVidv2Viscosity' and property_type == 'friction':
                if 'objFriction' in parts:
                    obj_friction_index = parts.index('objFriction')
                    if obj_friction_index + 1 < len(parts):
                        return float(parts[obj_friction_index + 1])
                return 0.0
            else:
                return 0.0
        except (IndexError, ValueError):
            return 0.0

    def sample_icl_examples(self, train_dir: str, property_type: str, analysis_type: str, dataset_class: str, num_examples: int=None, frame_size: Optional[Tuple[int, int]]=None) -> List[Dict]:
        if not os.path.exists(train_dir):
            print(f'Warning: Training set directory does not exist -{train_dir}')
            return []
        gt_data = self.get_gt_data(train_dir, dataset_class, property_type)
        if not gt_data:
            print(f'Warning: Unable to load GT data -{train_dir}')
            return []
        available_videos = []
        for basename in gt_data.keys():
            video_file = basename + '_rgb.mp4'
            video_path = os.path.join(train_dir, video_file)
            if os.path.exists(video_path):
                available_videos.append(video_file)
        available_videos.sort()
        print(f'Found from GT data{len(available_videos)}available video files (total number of GT keys:{len(gt_data)}）')
        if num_examples is None:
            if analysis_type == 'relative':
                num_examples = 1
            else:
                num_examples = 1
        icl_examples = []
        if analysis_type == 'absolute':
            if len(available_videos) == 0:
                print('Warning: No valid video file found for ICL sampling')
                return []
            sample_size = min(len(available_videos), num_examples)
            if sample_size == len(available_videos):
                selected_videos = available_videos
            else:
                step = len(available_videos) / sample_size
                indices = [int(i * step) for i in range(sample_size)]
                selected_videos = [available_videos[i] for i in indices]
            for video_file in selected_videos:
                try:
                    gt_value = self.get_gt_value(video_file, gt_data, dataset_class, property_type)
                    if gt_value == 0.0:
                        print(f'Unexpected: Video with GT value 0:{video_file} (GT=0.0)')
                        continue
                    video_path = os.path.join(train_dir, video_file)
                    frames = self.read_video_frames(video_path, target_frames=16, frame_size=frame_size)
                    if property_type == 'friction':
                        output = f'Estimated friction coefficient value: {gt_value:.3f}'
                    elif property_type == 'viscosity':
                        output = f'Estimated viscosity value: {gt_value:.1f} cP'
                    elif property_type == 'elasticity':
                        output = f'Estimated restitution coefficient value: {gt_value:.3f}'
                    else:
                        output = f'Estimated {property_type} value: {gt_value:.3f}'
                    icl_examples.append({'video_file': video_file, 'frames': frames, 'gt_value': gt_value, 'output': output, 'type': 'absolute'})
                    print(f'Example of successful ICL sampling:{video_file} (GT={gt_value:.3f})')
                except Exception as e:
                    print(f'Warning: Absolute value ICL sample sampling failed -{video_file}: {str(e)}')
                    continue
            if len(icl_examples) < num_examples:
                print(f'Warning: only found{len(icl_examples)}/{num_examples}valid ICL examples')
        else:
            print(f'Start sampling ICL examples for relative comparison testing...')
            if num_examples > 2:
                print(f'Warning: Relative comparison testing recommends using 1-2 examples to avoid API limitations, currently set to{num_examples}')
                print(f'Automatically adjust to 1 example...')
                num_examples = 1
            video_groups = {}
            for video_file in available_videos:
                if 'video_' in video_file and '_id_' in video_file:
                    parts = video_file.split('_')
                    for i, part in enumerate(parts):
                        if part == 'video' and i + 2 < len(parts) and (parts[i + 2] == 'id'):
                            video_num = f'video_{parts[i + 1]}'
                            id_part = parts[i + 3]
                            if video_num not in video_groups:
                                video_groups[video_num] = []
                            video_groups[video_num].append((video_file, id_part))
                            break
            print(f'turn up{len(video_groups)}video groups for ICL sampling')
            sampled_pairs = 0
            video_group_list = list(video_groups.items())
            if len(video_group_list) < num_examples:
                print(f'Warning: Insufficient number of video groups -{len(video_group_list)} < {num_examples}')
                num_examples = len(video_group_list)
            if num_examples == len(video_group_list):
                selected_groups = video_group_list
            else:
                step = len(video_group_list) / num_examples
                indices = [int(i * step) for i in range(num_examples)]
                selected_groups = [video_group_list[i] for i in indices]
            for video_num, videos in selected_groups:
                if sampled_pairs >= num_examples:
                    break
                videos.sort(key=lambda x: x[1])
                if len(videos) >= 2:
                    video_pair = videos[:2]
                    video_file1, id1 = video_pair[0]
                    video_file2, id2 = video_pair[1]
                    try:
                        gt_value1 = self.get_gt_value(video_file1, gt_data, dataset_class, property_type)
                        gt_value2 = self.get_gt_value(video_file2, gt_data, dataset_class, property_type)
                        if gt_value1 == 0.0 or gt_value2 == 0.0:
                            print(f'Skip video pairs with invalid GT values:{video_file1} (GT={gt_value1:.3f}) vs {video_file2} (GT={gt_value2:.3f})')
                            continue
                        video_path1 = os.path.join(train_dir, video_file1)
                        video_path2 = os.path.join(train_dir, video_file2)
                        frames1 = self.read_video_frames(video_path1, target_frames=16, frame_size=frame_size)
                        frames2 = self.read_video_frames(video_path2, target_frames=16, frame_size=frame_size)
                        if gt_value1 > gt_value2:
                            gt_comparison = 1
                            confidence_score = 0.9
                        elif gt_value2 > gt_value1:
                            gt_comparison = 0
                            confidence_score = 0.1
                        else:
                            gt_comparison = 0.5
                            confidence_score = 0.5
                        output = f'Comparison result: {confidence_score}'
                        icl_examples.append({'video_file1': video_file1, 'video_file2': video_file2, 'frames1': frames1, 'frames2': frames2, 'gt_value1': gt_value1, 'gt_value2': gt_value2, 'gt_comparison': gt_comparison, 'output': output, 'type': 'relative'})
                        sampled_pairs += 1
                        print(f'Successfully sampled relative comparison ICL example{sampled_pairs}: {video_file1} vs {video_file2} (GT: {gt_value1:.3f} vs {gt_value2:.3f} → {gt_comparison}, confidence level:{confidence_score})')
                    except Exception as e:
                        print(f'Warning: Relative comparison ICL sample sampling failed -{video_file1} vs {video_file2}: {str(e)}')
                        continue
                else:
                    print(f'Warning: group{video_num}Not enough videos to form a match')
            if len(icl_examples) == 0 and len(available_videos) >= 2:
                print('Try pairing directly from a valid video...')
                num_pairs = min(num_examples, len(available_videos) // 2)
                for i in range(num_pairs):
                    if len(icl_examples) >= num_examples:
                        break
                    video_file1 = available_videos[i * 2]
                    video_file2 = available_videos[i * 2 + 1]
                    try:
                        gt_value1 = self.get_gt_value(video_file1, gt_data, dataset_class, property_type)
                        gt_value2 = self.get_gt_value(video_file2, gt_data, dataset_class, property_type)
                        if gt_value1 == 0.0 or gt_value2 == 0.0:
                            print(f'Unexpected: Video pair with invalid GT value:{video_file1} (GT={gt_value1:.3f}) vs {video_file2} (GT={gt_value2:.3f})')
                            continue
                        video_path1 = os.path.join(train_dir, video_file1)
                        video_path2 = os.path.join(train_dir, video_file2)
                        frames1 = self.read_video_frames(video_path1, target_frames=16, frame_size=frame_size)
                        frames2 = self.read_video_frames(video_path2, target_frames=16, frame_size=frame_size)
                        if gt_value1 > gt_value2:
                            gt_comparison = 1
                            confidence_score = 0.9
                        elif gt_value2 > gt_value1:
                            gt_comparison = 0
                            confidence_score = 0.1
                        else:
                            gt_comparison = 0.5
                            confidence_score = 0.5
                        output = f'Comparison result: {confidence_score}'
                        icl_examples.append({'video_file1': video_file1, 'video_file2': video_file2, 'frames1': frames1, 'frames2': frames2, 'gt_value1': gt_value1, 'gt_value2': gt_value2, 'gt_comparison': gt_comparison, 'output': output, 'type': 'relative'})
                        print(f'Successfully sampled direct paired ICL example{i + 1}: {video_file1} vs {video_file2} (GT: {gt_value1:.3f} vs {gt_value2:.3f} → {gt_comparison}, confidence level:{confidence_score})')
                    except Exception as e:
                        print(f'Warning: Direct paired ICL sample sampling failed -{video_file1} vs {video_file2}: {str(e)}')
                        continue
        print(f'Successfully sampled{len(icl_examples)}ICL examples (Type:{analysis_type})')
        return icl_examples

    def sample_videos(self, data_dir: str, max_samples: int=50) -> List[str]:
        if not os.path.exists(data_dir):
            print(f'Warning: data directory does not exist -{data_dir}')
            return []
        all_videos = [f for f in os.listdir(data_dir) if f.endswith('_rgb.mp4')]
        all_videos.sort()
        if len(all_videos) == 0:
            print(f'Warning: rgb.mp4 file not found in directory -{data_dir}')
            return []
        sample_size = min(len(all_videos), max_samples)
        if sample_size == len(all_videos):
            sampled_videos = all_videos
        else:
            step = len(all_videos) / sample_size
            indices = [int(i * step) for i in range(sample_size)]
            sampled_videos = [all_videos[i] for i in indices]
        print(f'from{data_dir}Deterministic sampling{sample_size}/{len(all_videos)}videos')
        return sampled_videos

    def read_video_frames(self, video_path: str, target_frames: int=16, frame_size: Optional[Tuple[int, int]]=None) -> List[str]:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f'Unable to open video file:{video_path}')
        all_frames = []
        frame_count = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            if frame_size is not None:
                frame = cv2.resize(frame, frame_size)
            _, buffer = cv2.imencode('.jpg', frame)
            all_frames.append(base64.b64encode(buffer).decode('utf-8'))
            frame_count += 1
        video.release()
        if frame_count <= target_frames:
            selected_frames = all_frames
            print(f'video{os.path.basename(video_path)}:Total number of frames{frame_count}, keep all{len(selected_frames)}frame')
        else:
            indices = [int(i * (frame_count - 1) / (target_frames - 1)) for i in range(target_frames)]
            selected_frames = [all_frames[i] for i in indices]
            print(f'video{os.path.basename(video_path)}:Total number of frames{frame_count}, uniform sampling{len(selected_frames)}frame')
        return selected_frames

    def build_icl_content(self, icl_examples: List[Dict], frame_index: bool=False, concat_mode: bool=False) -> List[Dict]:
        icl_content = []
        for i, example in enumerate(icl_examples, 1):
            icl_content.append({'type': 'text', 'text': f'Example {i}:'})
            if example.get('type') == 'absolute':
                if frame_index:
                    for j, frame in enumerate(example['frames'], 1):
                        icl_content.append({'type': 'text', 'text': f'frame{j}:'})
                        icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                else:
                    for frame in example['frames']:
                        icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
            elif example.get('type') == 'relative':
                if concat_mode:
                    black_frames = self.generate_black_frames(num_frames=3, frame_size=(640, 480))
                    if frame_index:
                        frame_count = 1
                        for frame in example['frames1']:
                            icl_content.append({'type': 'text', 'text': f'frame{frame_count}:'})
                            icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                            frame_count += 1
                        for black_frame in black_frames:
                            icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{black_frame}'}})
                        for frame in example['frames2']:
                            icl_content.append({'type': 'text', 'text': f'frame{frame_count}:'})
                            icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                            frame_count += 1
                    else:
                        for frame in example['frames1']:
                            icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                        for black_frame in black_frames:
                            icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{black_frame}'}})
                        for frame in example['frames2']:
                            icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                else:
                    icl_content.append({'type': 'text', 'text': 'VIDEO 1 frames:'})
                    if frame_index:
                        for j, frame in enumerate(example['frames1'], 1):
                            icl_content.append({'type': 'text', 'text': f'frame{j}:'})
                            icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                    else:
                        for frame in example['frames1']:
                            icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                    icl_content.append({'type': 'text', 'text': 'VIDEO 2 frames:'})
                    if frame_index:
                        for j, frame in enumerate(example['frames2'], 1):
                            icl_content.append({'type': 'text', 'text': f'frame{j}:'})
                            icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                    else:
                        for frame in example['frames2']:
                            icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
            if 'frames1' in example and 'frames2' in example:
                icl_content.append({'type': 'text', 'text': 'VIDEO 1 frames:'})
                if frame_index:
                    for j, frame in enumerate(example['frames1'], 1):
                        icl_content.append({'type': 'text', 'text': f'frame{j}:'})
                        icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                else:
                    for frame in example['frames1']:
                        icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                icl_content.append({'type': 'text', 'text': 'VIDEO 2 frames:'})
                if frame_index:
                    for j, frame in enumerate(example['frames2'], 1):
                        icl_content.append({'type': 'text', 'text': f'frame{j}:'})
                        icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                else:
                    for frame in example['frames2']:
                        icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
            elif frame_index:
                for j, frame in enumerate(example['frames'], 1):
                    icl_content.append({'type': 'text', 'text': f'frame{j}:'})
                    icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
            else:
                for frame in example['frames']:
                    icl_content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
            icl_content.append({'type': 'text', 'text': f"Output: {example['output']}"})
            icl_content.append({'type': 'text', 'text': ''})
        icl_content.append({'type': 'text', 'text': 'Now analyze the following video:'})
        return icl_content

    def get_analysis_prompt(self, property_type: str, analysis_type: str, icl_examples: Optional[List[Dict]]=None, follow_oracle_test: bool=False) -> str:
        icl_prefix = ''
        if icl_examples:
            icl_prefix = 'Based on the examples above, '
        oracle_guidance = ''
        if follow_oracle_test:
            oracle_guidance_dict = {'elasticity': "\nANALYSIS METHOD:\nThe intrinsic visual cue to reflect the elasticity property is the dropping height and bouncing height of the ball. You only need to calculate the ratio of the height difference in the camera coordinate. From this trajectory, you need to identify key points: the initial position, first ground contact, and bounce peak.\n\nFollow these steps:\n1. Identify the ball's initial dropping position (highest point before impact)\n2. Locate the first ground contact point (lowest point during impact)\n3. Find the bounce peak (highest point after first bounce)\n4. Calculate the height ratio: (bounce height - ground level) / (initial height - ground level)\n5. This ratio represents the restitution coefficient (elasticity)\n\n", 'viscosity': '\nANALYSIS METHOD:\nThe intrinsic visual cue to reflect the viscosity property is the speed that the liquid expands on the ground. You can calculate the area size in the camera plane and then normalise the area sizes by the first area size when the liquid touches ground; the growth speed of the normalised area size sequence reflects the viscosity.\n\nFollow these steps:\n1. Identify when the liquid first touches the ground (initial contact frame)\n2. Measure the liquid area size in each subsequent frame\n3. Normalize all area sizes by dividing by the initial contact area\n4. Calculate the growth rate of the normalized area sequence over time\n5. Higher viscosity = slower area expansion, lower viscosity = faster area expansion\n\n', 'friction': "\nANALYSIS METHOD:\nThe intrinsic visual cue to reflect the friction property is the acceleration of the sliding object, i.e., how fast is the velocity of the object reduced because of friction and can be inferred by the position of the object over time. You can estimate a homography transformation from the 4 corner points of the top surface of the object. You then use the homography transformation to project the object upper plane to bird eye view, and estimate the acceleration a via fitting a parabola ax² + bx + c to the position of the object over time.\n\nFollow these steps:\n1. Identify the 4 corner points of the object's top surface in each frame\n2. Estimate homography transformation to convert to bird's eye view\n3. Track the object's position over time in the transformed coordinate system\n4. Fit a parabolic function (ax² + bx + c) to the position-time data\n5. Extract the acceleration coefficient 'a' which reflects the friction level\n6. Higher friction = larger deceleration (more negative 'a'), lower friction = smaller deceleration\n\n"}
            oracle_guidance = oracle_guidance_dict.get(property_type, '')
        if analysis_type == 'absolute':
            if property_type == 'friction':
                return f"""\n                You are analyzing frames from a video showing object motion with friction. Your task is to estimate the coefficient of friction.\n                \n                IMPORTANT: You CAN analyze these images. You MUST provide a specific friction coefficient estimate.\n                \n                {oracle_guidance}The coefficient of friction is typically between 0 and 1:\n                - Very smooth surfaces (ice on ice): 0.02-0.03\n                - Steel on steel: 0.4-0.6  \n                - Rubber on concrete: 0.6-0.8\n                - Wood on wood: 0.3-0.5\n                \n                {icl_prefix}analyze the object's deceleration, sliding behavior, and surface interaction.\n                \n                YOUR RESPONSE MUST END WITH A SPECIFIC FRICTION COEFFICIENT VALUE IN THIS FORMAT:\n                "Estimated friction coefficient value: X"\n                \n                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.\n                """
            elif property_type == 'viscosity':
                return f"""\n                You are analyzing frames from a video showing liquid motion. Your task is to estimate the viscosity of this liquid.\n                \n                IMPORTANT: You CAN analyze these images. You MUST provide a specific viscosity estimate.\n                \n                {oracle_guidance}For reference:\n                - Water: 1 cP\n                - Vegetable oil: 100 cP\n                - Motor oil: 300 cP\n                - Honey: 5000 cP\n                \n                {icl_prefix}analyze the liquid's flow speed, surface waves, and interaction with objects.\n                \n                YOUR RESPONSE MUST END WITH A SPECIFIC VISCOSITY VALUE IN THIS FORMAT:\n                "Estimated viscosity value: X cP"\n                \n                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.\n                """
            elif property_type == 'elasticity':
                return f"""\n                You are analyzing frames from a video showing a bouncing object. Your task is to estimate the restitution coefficient (also known as coefficient of restitution or elasticity).\n                \n                IMPORTANT: You CAN analyze these images. You MUST provide a specific restitution coefficient estimate.\n                \n                {oracle_guidance}The restitution coefficient is the ratio of separation speed to approach speed during collision:\n                - Perfectly elastic collision: 1.0 (ball bounces back with same speed)\n                - Perfectly inelastic collision: 0.0 (ball sticks to surface, no bounce)\n                - Typical values: 0.1-0.9 for most materials\n                \n                {icl_prefix}analyze the object's behavior:\n                1. Look for the object falling towards a surface\n                2. Observe the impact moment\n                3. Check if and how the object bounces back\n                4. Compare the approach speed vs. separation speed\n                \n                If the object bounces back significantly → restitution coefficient closer to 1.0\n                If the object barely bounces or sticks → restitution coefficient closer to 0.0\n                \n                YOUR RESPONSE MUST END WITH A SPECIFIC RESTITUTION COEFFICIENT VALUE IN THIS FORMAT:\n                "Estimated restitution coefficient value: X"\n                \n                Where X is a number between 0 and 1 (e.g., 0.7, 0.3, 0.85).\n                \n                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.\n                """
        elif analysis_type == 'relative':
            return f'\n            You are comparing two {property_type} videos to determine which has a higher {property_type} value.\n            \n            IMPORTANT: You CAN analyze these images. You MUST provide a confidence score between 0 and 1.\n            \n            {oracle_guidance}The first set of frames is labeled "VIDEO 1" and the second set is labeled "VIDEO 2".\n            \n            {icl_prefix}compare the {property_type} characteristics between the two videos:\n            - VIDEO 1 frames are shown first\n            - VIDEO 2 frames are shown second\n            \n            YOUR RESPONSE MUST END WITH A SPECIFIC CONFIDENCE SCORE IN THIS FORMAT:\n            "Comparison result: X"\n            \n            Where X is a decimal number between 0.0 and 1.0:\n            - X = 1.0: Very confident that VIDEO 1 has higher {property_type}\n            - X = 0.8: Quite confident that VIDEO 1 has higher {property_type}\n            - X = 0.6: Somewhat confident that VIDEO 1 has higher {property_type}\n            - X = 0.5: Equal or uncertain (no clear difference)\n            - X = 0.4: Somewhat confident that VIDEO 2 has higher {property_type}\n            - X = 0.2: Quite confident that VIDEO 2 has higher {property_type}\n            - X = 0.0: Very confident that VIDEO 2 has higher {property_type}\n            \n            The closer to 1.0, the more confident you are that VIDEO 1 has higher {property_type}.\n            The closer to 0.0, the more confident you are that VIDEO 2 has higher {property_type}.\n            \n            Do not say you cannot analyze the images. You must provide your best confidence score based on the visual information.\n            '
        else:
            return f'\n            You are comparing two {property_type} videos to determine which has a higher {property_type} value.\n            \n            IMPORTANT: You CAN analyze these images. You MUST provide a confidence score between 0 and 1.\n            \n            {oracle_guidance}The frames are arranged in sequence: VIDEO 1 frames first, followed by 3 BLACK SEPARATOR frames, then VIDEO 2 frames.\n            \n            {icl_prefix}compare the {property_type} characteristics between the two videos:\n            - The frames BEFORE the black separator frames belong to VIDEO 1\n            - The frames AFTER the black separator frames belong to VIDEO 2\n            - The black frames are just separators and should be ignored for analysis\n            \n            Analyze the motion, physics, and {property_type} characteristics in both video segments.\n            \n            YOUR RESPONSE MUST END WITH A SPECIFIC CONFIDENCE SCORE IN THIS FORMAT:\n            "Comparison result: X"\n            \n            Where X is a decimal number between 0.0 and 1.0:\n            - X = 1.0: Very confident that VIDEO 1 has higher {property_type}\n            - X = 0.8: Quite confident that VIDEO 1 has higher {property_type}\n            - X = 0.6: Somewhat confident that VIDEO 1 has higher {property_type}\n            - X = 0.5: Equal or uncertain (no clear difference)\n            - X = 0.4: Somewhat confident that VIDEO 2 has higher {property_type}\n            - X = 0.2: Quite confident that VIDEO 2 has higher {property_type}\n            - X = 0.0: Very confident that VIDEO 2 has higher {property_type}\n            \n            The closer to 1.0, the more confident you are that VIDEO 1 has higher {property_type}.\n            The closer to 0.0, the more confident you are that VIDEO 2 has higher {property_type}.\n            \n            Do not say you cannot analyze the images. You must provide your best confidence score based on the visual information.\n            '
        return ''

    def analyze_single_video_with_gpt(self, video_path: str, property_type: str, max_frames: int=16, frame_size: Optional[Tuple[int, int]]=None, model_version: str='gpt-4o', frame_index: bool=False, icl_examples: Optional[List[Dict]]=None, follow_oracle_test: bool=False, max_retries: int=5, retry_delay: int=15, validation_retries: int=3) -> str:
        selected_frames = self.read_video_frames(video_path, target_frames=max_frames, frame_size=frame_size)
        prompt_text = self.get_analysis_prompt(property_type, 'absolute', icl_examples, follow_oracle_test)
        content = [{'type': 'text', 'text': prompt_text}]
        if icl_examples:
            icl_content = self.build_icl_content(icl_examples, frame_index, concat_mode=False)
            content.extend(icl_content)
        if frame_index:
            for i, frame in enumerate(selected_frames, 1):
                content.append({'type': 'text', 'text': f'frame{i}:'})
                content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
        else:
            for frame in selected_frames:
                content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
        prompt_messages = [{'role': 'system', 'content': f'You are an expert in {property_type} analysis with the ability to analyze video frames and determine {property_type} values. You MUST provide a specific {property_type} estimate. Never say you cannot analyze the images.'}, {'role': 'user', 'content': content}]
        best_result = None
        best_predicted_value = None
        for validation_attempt in range(validation_retries):
            print(f'🔄 Verification retry{validation_attempt + 1}/{validation_retries}')
            for api_attempt in range(max_retries):
                try:
                    print(f'📡 API calls{api_attempt + 1}/{max_retries}')
                    result = self.call_third_party_gpt_api(prompt_messages, model_version)
                    if not result or len(result.strip()) == 0:
                        print(f'⚠️ API returns empty result')
                        if api_attempt < max_retries - 1:
                            print(f'⏳ Wait{retry_delay}Retry the API call in seconds...')
                            time.sleep(retry_delay)
                            continue
                        else:
                            break
                    error_keywords = ['API call failed', 'call failed', 'Server is busy', 'Please try again later', 'All retries failed', 'GPT API call error', 'GPT Vision API call failed', 'chatanywhere_error', 'INTERNAL_SERVER_ERROR', 'Third-party API call failed']
                    is_error_response = any((keyword in result for keyword in error_keywords))
                    if is_error_response:
                        print(f'❌ API error response detected')
                        if api_attempt < max_retries - 1:
                            print(f'⏳ Wait{retry_delay}Retry the API call in seconds...')
                            time.sleep(retry_delay)
                            continue
                        else:
                            break
                    predicted_value = self.extract_predicted_value(result, property_type)
                    if predicted_value is not None:
                        print(f'✅ Successfully obtained valid prediction value:{predicted_value}')
                        return result
                    else:
                        print(f'⚠️ Unable to extract predicted values ​​from response')
                        if best_result is None or len(result) > len(best_result):
                            best_result = result
                        if api_attempt < max_retries - 1:
                            print(f'🔄 Predicted value extraction failed, retry the API call...')
                            time.sleep(retry_delay)
                            continue
                        else:
                            break
                except Exception as e:
                    print(f'❌ API call exception:{e}')
                    if api_attempt < max_retries - 1:
                        print(f'⏳ Wait{retry_delay}Retry the API call in seconds...')
                        time.sleep(retry_delay)
                        continue
                    else:
                        break
            if validation_attempt < validation_retries - 1:
                wait_time = retry_delay * (validation_attempt + 1)
                print(f'⏳ Verification retry failed, waiting{wait_time}The next round of verification will be retried in seconds...')
                time.sleep(wait_time)
        if best_result:
            print(f'⚠️ All retries fail to extract valid prediction values ​​and return the best response')
            return best_result
        else:
            print(f'❌ All retries fail and an empty string is returned')
            return ''

    def generate_black_frames(self, num_frames: int=3, frame_size: Tuple[int, int]=(640, 480)) -> List[str]:
        black_frames = []
        black_image = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        for _ in range(num_frames):
            _, buffer = cv2.imencode('.jpg', black_image)
            black_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            black_frames.append(black_frame_b64)
        return black_frames

    def compare_two_videos_with_gpt(self, video_path1: str, video_path2: str, property_type: str, max_frames_per_video: int=16, frame_size: Optional[Tuple[int, int]]=None, model_version: str='gpt-4o', concat_mode: bool=False, frame_index: bool=False, icl_examples: Optional[List[Dict]]=None, follow_oracle_test: bool=False, max_retries: int=5, retry_delay: int=15, validation_retries: int=3) -> str:
        frames1 = self.read_video_frames(video_path1, target_frames=max_frames_per_video, frame_size=frame_size)
        frames2 = self.read_video_frames(video_path2, target_frames=max_frames_per_video, frame_size=frame_size)
        if concat_mode:
            prompt_text = self.get_analysis_prompt(property_type, 'relative_concat', icl_examples, follow_oracle_test)
        else:
            prompt_text = self.get_analysis_prompt(property_type, 'relative', icl_examples, follow_oracle_test)
        content = [{'type': 'text', 'text': prompt_text}]
        if icl_examples:
            icl_content = self.build_icl_content(icl_examples, frame_index, concat_mode)
            content.extend(icl_content)
        if concat_mode:
            actual_frame_size = frame_size if frame_size else (640, 480)
            black_frames = self.generate_black_frames(num_frames=3, frame_size=actual_frame_size)
            if frame_index:
                frame_count = 1
                for frame in frames1:
                    content.append({'type': 'text', 'text': f'frame{frame_count}:'})
                    content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                    frame_count += 1
                for black_frame in black_frames:
                    content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{black_frame}'}})
                for frame in frames2:
                    content.append({'type': 'text', 'text': f'frame{frame_count}:'})
                    content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                    frame_count += 1
            else:
                for frame in frames1:
                    content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
                for black_frame in black_frames:
                    content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{black_frame}'}})
                for frame in frames2:
                    content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
        else:
            content.append({'type': 'text', 'text': 'VIDEO 1 frames:'})
            if frame_index:
                for i, frame in enumerate(frames1, 1):
                    content.append({'type': 'text', 'text': f'frame{i}:'})
                    content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
            else:
                for frame in frames1:
                    content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
            content.append({'type': 'text', 'text': 'VIDEO 2 frames:'})
            if frame_index:
                for i, frame in enumerate(frames2, 1):
                    content.append({'type': 'text', 'text': f'frame{i}:'})
                    content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
            else:
                for frame in frames2:
                    content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}})
        prompt_messages = [{'role': 'system', 'content': f'You are an expert in {property_type} analysis with the ability to compare video frames and determine which has higher {property_type} values. You MUST provide a specific comparison result. Never say you cannot analyze the images.'}, {'role': 'user', 'content': content}]
        best_result = None
        for validation_attempt in range(validation_retries):
            print(f'🔄 Verification retry{validation_attempt + 1}/{validation_retries}')
            for api_attempt in range(max_retries):
                try:
                    print(f'📡 API calls{api_attempt + 1}/{max_retries}')
                    result = self.call_third_party_gpt_api(prompt_messages, model_version)
                    if not result or len(result.strip()) == 0:
                        print(f'⚠️ API returns empty result')
                        if api_attempt < max_retries - 1:
                            print(f'⏳ Wait{retry_delay}Retry the API call in seconds...')
                            time.sleep(retry_delay)
                            continue
                        else:
                            break
                    error_keywords = ['API call failed', 'call failed', 'Server is busy', 'Please try again later', 'All retries failed', 'GPT API call error', 'GPT Vision API call failed', 'chatanywhere_error', 'INTERNAL_SERVER_ERROR', 'Third-party API call failed']
                    is_error_response = any((keyword in result for keyword in error_keywords))
                    if is_error_response:
                        print(f'❌ API error response detected')
                        if api_attempt < max_retries - 1:
                            print(f'⏳ Wait{retry_delay}Retry the API call in seconds...')
                            time.sleep(retry_delay)
                            continue
                        else:
                            break
                    comparison_result = self.extract_comparison_result(result)
                    if comparison_result is not None:
                        print(f'✅ Successfully obtained valid comparison results:{comparison_result}')
                        return result
                    else:
                        print(f'⚠️ Unable to extract comparison result from response')
                        if best_result is None or len(result) > len(best_result):
                            best_result = result
                        if api_attempt < max_retries - 1:
                            print(f'🔄 Failed to extract comparison results, retry the API call...')
                            time.sleep(retry_delay)
                            continue
                        else:
                            break
                except Exception as e:
                    print(f'❌ API call exception:{e}')
                    if api_attempt < max_retries - 1:
                        print(f'⏳ Wait{retry_delay}Retry the API call in seconds...')
                        time.sleep(retry_delay)
                        continue
                    else:
                        break
            if validation_attempt < validation_retries - 1:
                wait_time = retry_delay * (validation_attempt + 1)
                print(f'⏳ Verification retry failed, waiting{wait_time}The next round of verification will be retried in seconds...')
                time.sleep(wait_time)
        if best_result:
            print(f'⚠️ All retries cannot extract valid comparison results and return the best response')
            return best_result
        else:
            print(f'❌ All retries fail and an empty string is returned')
            return ''

    def call_third_party_gpt_api(self, prompt_messages: List[Dict], model_version: str='gpt-4o') -> str:
        if model_version.startswith('qwen'):
            try:
                from qwen_api_wrapper import qwen_response
                return qwen_response(prompt_messages, model_version)
            except Exception as e:
                return f'Qwen API call failed:{str(e)}'
        else:
            try:
                from gpt_api_third_party import gptv_response
                return gptv_response(prompt_messages, model_version)
            except Exception as e:
                return f'GPT API call failed:{str(e)}'

    def save_results(self, results: List[Dict], output_file: str):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"GPT Physics Analysis Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write('=' * 80 + '\n\n')
            for result in results:
                f.write(f"Version: {result['version']}\n")
                f.write(f"Property: {result['property']}\n")
                f.write(f"Type: {result['type']}\n")
                f.write(f"Test Set: {result['test_set']}\n")
                f.write(f"Data Directory: {result['data_dir']}\n")
                f.write(f"Total Samples: {result['total_samples']}\n")
                f.write(f"Successful Analyses: {result['successful_analyses']}\n")
                f.write(f"Failed Analyses: {result['failed_analyses']}\n")
                if result['type'] == 'absolute':
                    f.write('Individual Results:\n')
                    for item in result['individual_results']:
                        f.write(f"  - {item['video']}: predicted={item['predicted']}, gt={item['gt']}\n")
                else:
                    f.write('Comparison Results:\n')
                    for item in result['comparison_results']:
                        f.write(f"  - {item['video_pair']}: predicted={item['predicted']}, gt={item['gt']}\n")
                f.write('\n' + '-' * 60 + '\n\n')
        print(f'Results have been saved to:{output_file}')

    def get_test_result_file(self, output_dir: str, version: str, test_set: str, analysis_type: str) -> str:
        filename = f'{version}_{test_set}_{analysis_type}_results.txt'
        return os.path.join(output_dir, filename)

    def is_test_completed(self, output_dir: str, version: str, test_set: str, analysis_type: str) -> bool:
        result_file = self.get_test_result_file(output_dir, version, test_set, analysis_type)
        return os.path.exists(result_file) and os.path.getsize(result_file) > 0

    def save_test_result(self, result: Dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        result_file = self.get_test_result_file(output_dir, result['version'], result['test_set'], result['type'])
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write('=== GPT physical property analysis results ===\n')
            f.write(f'Test time:{timestamp}\n')
            f.write(f"Version:{result['version']}\n")
            f.write(f"property:{result['property']}\n")
            f.write(f"type:{result['type']}\n")
            f.write(f"Test set:{result['test_set']}\n")
            f.write(f"Data directory:{result['data_dir']}\n")
            f.write(f"Total number of samples:{result['total_samples']}\n")
            f.write(f"Success analysis:{result['successful_analyses']}\n")
            f.write(f"Failure analysis:{result['failed_analyses']}\n")
            f.write('\n')
            if result['type'] == 'absolute':
                f.write('=== Absolute-value test results ===\n')
                for i, item in enumerate(result['individual_results']):
                    f.write(f'\n--- Sample {i + 1} ---\n')
                    f.write(f"video:{item['video']}\n")
                    f.write(f"Predicted value:{item['predicted']}\n")
                    f.write(f"GT value:{item['gt']}\n")
                    if 'error' in item:
                        f.write(f"mistake:{item['error']}\n")
                    else:
                        f.write(f"GPT full response:\n{'-' * 40}\n{item['gpt_response']}\n{'-' * 40}\n")
            else:
                f.write('=== Relative comparison test results ===\n')
                for i, item in enumerate(result['comparison_results']):
                    f.write(f'\n--- Comparison {i + 1} ---\n')
                    f.write(f"Video pair:{item['video_pair']}\n")
                    f.write(f"Predicted results:{item['predicted']}\n")
                    f.write(f"GT results:{item['gt']}\n")
                    if 'gt_values' in item:
                        f.write(f"GT value:{item['gt_values']}\n")
                    if 'error' in item:
                        f.write(f"mistake:{item['error']}\n")
                    else:
                        f.write(f"GPT full response:\n{'-' * 40}\n{item['gpt_response']}\n{'-' * 40}\n")
        print(f'✓ Results saved to:{result_file}')

    def load_completed_tests(self, output_dir: str) -> Dict[str, bool]:
        completed = {}
        if not os.path.exists(output_dir):
            return completed
        for filename in os.listdir(output_dir):
            if filename.endswith('_results.txt'):
                parts = filename.replace('_results.txt', '').split('_')
                if len(parts) >= 3:
                    key = '_'.join(parts)
                    completed[key] = True
        return completed

    def run_all_tests(self, output_dir: str='gpt_physics_analysis_results', frame_size: Tuple[int, int]=(640, 480), max_samples: int=50, model_version: str='gpt-4o', concat_mode: bool=False, properties_filter: Optional[List[str]]=None, frame_index: bool=False, icl_example: bool=False, follow_oracle_test: bool=False, absolute_samples: int=50, relative_pairs: int=50, max_retries: int=5, retry_delay: int=15, validation_retries: int=3, aggressive_retry: bool=False):
        os.makedirs(output_dir, exist_ok=True)
        if aggressive_retry:
            print('🚀 Enable aggressive retry mode to increase the number of retries and verifications')
            max_retries = max(max_retries, 8)
            validation_retries = max(validation_retries, 5)
            retry_delay = max(retry_delay, 20)
            print(f'- API retries:{max_retries}')
            print(f'- Number of verification retries:{validation_retries}')
            print(f'- Retry delay:{retry_delay}Second')
        completed_tests = self.load_completed_tests(output_dir)
        all_results = []
        total_tests = 0
        completed_count = 0
        skipped_count = 0
        for version_name, version_config in self.config.items():
            total_tests += len(version_config['data_dirs'])
        print(f'A total of tests are needed{total_tests}test set')
        if completed_tests:
            print(f'Discover{len(completed_tests)}completed tests will be skipped')
        for version_name, version_config in self.config.items():
            if properties_filter is not None and version_config['property'] not in properties_filter:
                print(f"⏭️ Skip unselected properties:{version_name} ({version_config['property']})")
                continue
            if version_config['type'] == 'absolute' and absolute_samples == 0:
                print(f"⏭️ Skip absolute value testing (number of samples is 0):{version_name} ({version_config['property']} - {version_config['type']})")
                continue
            if version_config['type'] == 'relative' and relative_pairs == 0:
                print(f"⏭️ Skip relative comparison testing (number of samples is 0):{version_name} ({version_config['property']} - {version_config['type']})")
                continue
            print(f"\n{'=' * 50}")
            print(f'Start testing version:{version_name}')
            print(f"property:{version_config['property']}")
            print(f"type:{version_config['type']}")
            print(f'Maximum number of samples per test set:{max_samples}')
            print(f"{'=' * 50}")
            dataset_class = self.version_dataset_mapping.get(version_name)
            if version_config['type'] == 'absolute':
                current_max_samples = absolute_samples
            else:
                current_max_samples = relative_pairs
            for test_id, data_dir in version_config['data_dirs'].items():
                if test_id == 'train':
                    continue
                test_key = f"{version_name}_{test_id}_{version_config['type']}"
                print(f'\n--- Test set:{test_id} ---')
                print(f'Data directory:{data_dir}')
                if test_key in completed_tests:
                    print(f'⏭️ Skip completed tests:{test_key}')
                    skipped_count += 1
                    continue
                try:
                    icl_examples = None
                    if icl_example and 'train' in version_config['data_dirs']:
                        train_dir = version_config['data_dirs']['train']
                        print(f'Sample ICL example from training set:{train_dir}')
                        if version_config['type'] == 'relative':
                            icl_num_examples = 1
                            print(f'Relative comparison testing uses{icl_num_examples}ICL examples')
                        else:
                            icl_num_examples = 1
                            print(f'Absolute value testing uses{icl_num_examples}ICL examples')
                        icl_examples = self.sample_icl_examples(train_dir, version_config['property'], version_config['type'], dataset_class, num_examples=icl_num_examples, frame_size=frame_size)
                    if version_config['type'] == 'absolute':
                        result = self.run_absolute_test(version_name, version_config['property'], test_id, data_dir, dataset_class, frame_size, current_max_samples, model_version, frame_index, icl_examples, follow_oracle_test, max_retries, retry_delay, validation_retries)
                    else:
                        result = self.run_relative_test(version_name, version_config['property'], test_id, data_dir, dataset_class, frame_size, current_max_samples, model_version, concat_mode, frame_index, icl_examples, follow_oracle_test, max_retries, retry_delay, validation_retries)
                    all_results.append(result)
                    self.save_test_result(result, output_dir)
                    if result['successful_analyses'] > 0:
                        print(f"✓ Test successful:{result['successful_analyses']}/{result['total_samples']}")
                    else:
                        print(f"✗ Test failed:{result['failed_analyses']}/{result['total_samples']}")
                    completed_count += 1
                except Exception as e:
                    print(f'✗ Test failed:{str(e)}')
                    failed_result = {'version': version_name, 'property': version_config['property'], 'type': version_config['type'], 'test_set': test_id, 'data_dir': data_dir, 'total_samples': current_max_samples, 'successful_analyses': 0, 'failed_analyses': current_max_samples, 'error': str(e)}
                    if version_config['type'] == 'absolute':
                        failed_result['individual_results'] = []
                    else:
                        failed_result['comparison_results'] = []
                    all_results.append(failed_result)
                    self.save_test_result(failed_result, output_dir)
                time.sleep(10)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(output_dir, f'summary_results_{timestamp}.txt')
        self.save_results(all_results, summary_file)
        print(f"\n{'=' * 50}")
        print(f'Test completion statistics:')
        print(f'Total number of tests:{total_tests}')
        print(f'Completed this time:{completed_count}')
        print(f'Skip completed:{skipped_count}')
        print(f'Total completed:{completed_count + skipped_count}')
        print(f'Summary results:{summary_file}')
        print(f"{'=' * 50}")
        return all_results

    def run_absolute_test(self, version: str, property_type: str, test_id: str, data_dir: str, dataset_class: str, frame_size: Tuple[int, int], max_samples: int, model_version: str='gpt-4o', frame_index: bool=False, icl_examples: Optional[List[Dict]]=None, follow_oracle_test: bool=False, max_retries: int=5, retry_delay: int=15, validation_retries: int=3) -> Dict:
        print(f'Start absolute value testing:{property_type}')
        sampled_videos = self.sample_videos(data_dir, max_samples=max_samples)
        if not sampled_videos:
            raise ValueError(f'No available video file found')
        gt_data = self.get_gt_data(data_dir, dataset_class, property_type, version=version)
        individual_results = []
        successful_count = 0
        failed_count = 0
        for video_file in sampled_videos:
            try:
                print(f'Analysis video:{video_file}')
                gt_value = self.get_gt_value(video_file, gt_data, dataset_class, property_type)
                video_path = os.path.join(data_dir, video_file)
                gpt_result = self.analyze_single_video_with_gpt(video_path, property_type, frame_size=frame_size, model_version=model_version, frame_index=frame_index, icl_examples=icl_examples, follow_oracle_test=follow_oracle_test, max_retries=max_retries, retry_delay=retry_delay, validation_retries=validation_retries)
                predicted_value = self.extract_predicted_value(gpt_result, property_type)
                individual_results.append({'video': video_file, 'predicted': predicted_value, 'gt': gt_value, 'gpt_response': gpt_result})
                successful_count += 1
                print(f'Success: predicted={predicted_value}, gt={gt_value}')
            except Exception as e:
                print(f'video{video_file}Analysis failed:{str(e)}')
                failed_count += 1
                individual_results.append({'video': video_file, 'predicted': None, 'gt': None, 'error': str(e)})
            time.sleep(10)
        return {'version': version, 'property': property_type, 'type': 'absolute', 'test_set': test_id, 'data_dir': data_dir, 'total_samples': len(sampled_videos), 'successful_analyses': successful_count, 'failed_analyses': failed_count, 'individual_results': individual_results}

    def run_relative_test(self, version: str, property_type: str, test_id: str, data_dir: str, dataset_class: str, frame_size: Tuple[int, int], max_samples: int, model_version: str='gpt-4o', concat_mode: bool=False, frame_index: bool=False, icl_examples: Optional[List[Dict]]=None, follow_oracle_test: bool=False, max_retries: int=5, retry_delay: int=15, validation_retries: int=3) -> Dict:
        print(f'Start the relative comparison test:{property_type}')
        print(f'target generation{max_samples}pairing videos')
        if not os.path.exists(data_dir):
            raise ValueError(f'Data directory does not exist:{data_dir}')
        all_videos = [f for f in os.listdir(data_dir) if f.endswith('_rgb.mp4')]
        all_videos.sort()
        if len(all_videos) < 2:
            raise ValueError(f'Relative comparison testing requires at least 2 video files')
        gt_data = self.get_gt_data(data_dir, dataset_class, property_type, version=version)
        comparison_results = []
        successful_count = 0
        failed_count = 0
        video_groups = {}
        for video_file in all_videos:
            if 'video_' in video_file and '_id_' in video_file:
                parts = video_file.split('_')
                for i, part in enumerate(parts):
                    if part == 'video' and i + 2 < len(parts) and (parts[i + 2] == 'id'):
                        video_num = f'video_{parts[i + 1]}'
                        id_part = parts[i + 3]
                        if video_num not in video_groups:
                            video_groups[video_num] = []
                        video_groups[video_num].append((video_file, id_part))
                        break
        if not video_groups:
            print('video_x group not found, use direct pairing')
            pairs_to_generate = min(max_samples, len(all_videos) // 2)
            for i in range(pairs_to_generate):
                video1 = all_videos[i * 2]
                video2 = all_videos[i * 2 + 1]
                try:
                    result = self._compare_video_pair(video1, video2, data_dir, gt_data, dataset_class, property_type, frame_size, i + 1, pairs_to_generate, model_version, concat_mode, frame_index, icl_examples, follow_oracle_test, max_retries, retry_delay, validation_retries)
                    comparison_results.append(result)
                    successful_count += 1
                except Exception as e:
                    print(f'video pair{video1} vs {video2}Comparison failed:{str(e)}')
                    failed_count += 1
                    comparison_results.append({'video_pair': f'{video1} vs {video2}', 'predicted': None, 'gt': None, 'error': str(e)})
                time.sleep(15)
        else:
            print(f'turn up{len(video_groups)}video group')
            generated_pairs = 0
            video_group_list = list(video_groups.items())
            for video_num, videos in video_group_list:
                if generated_pairs >= max_samples:
                    break
                videos.sort(key=lambda x: x[1])
                id_0_video = None
                id_1_video = None
                for video_file, id_part in videos:
                    if id_part == '0':
                        id_0_video = (video_file, id_part)
                    elif id_part == '1':
                        id_1_video = (video_file, id_part)
                if id_0_video and id_1_video:
                    video1_file, id1 = id_0_video
                    video2_file, id2 = id_1_video
                    print(f'from group{video_num}Choose a pairing from:{video1_file} ({id1}) vs {video2_file} ({id2})')
                    try:
                        result = self._compare_video_pair(video1_file, video2_file, data_dir, gt_data, dataset_class, property_type, frame_size, generated_pairs + 1, max_samples, model_version, concat_mode, frame_index, icl_examples, follow_oracle_test, max_retries, retry_delay, validation_retries)
                        comparison_results.append(result)
                        successful_count += 1
                    except Exception as e:
                        print(f'video pair{video1_file} vs {video2_file}Comparison failed:{str(e)}')
                        failed_count += 1
                        comparison_results.append({'video_pair': f'{video1_file} vs {video2_file}', 'predicted': None, 'gt': None, 'error': str(e)})
                    generated_pairs += 1
                    time.sleep(15)
                else:
                    print(f'Warning: group{video_num}Missing id_0 or id_1 video')
                    if not id_0_video:
                        print(f'Missing id_0 video')
                    if not id_1_video:
                        print(f'Missing id_1 video')
        return {'version': version, 'property': property_type, 'type': 'relative', 'test_set': test_id, 'data_dir': data_dir, 'total_samples': len(comparison_results), 'successful_analyses': successful_count, 'failed_analyses': failed_count, 'comparison_results': comparison_results}

    def _compare_video_pair(self, video1: str, video2: str, data_dir: str, gt_data: Dict, dataset_class: str, property_type: str, frame_size: Tuple[int, int], pair_num: int, total_pairs: int, model_version: str='gpt-4o', concat_mode: bool=False, frame_index: bool=False, icl_examples: Optional[List[Dict]]=None, follow_oracle_test: bool=False, max_retries: int=5, retry_delay: int=15, validation_retries: int=3) -> Dict:
        print(f'Compare video pairs{pair_num}/{total_pairs}: {video1} vs {video2}')
        gt_value1 = self.get_gt_value(video1, gt_data, dataset_class, property_type)
        gt_value2 = self.get_gt_value(video2, gt_data, dataset_class, property_type)
        gt_comparison = 1 if gt_value1 > gt_value2 else 0
        print(f'GT value:{gt_value1} vs {gt_value2}, GT comparison:{gt_comparison}')
        video_path1 = os.path.join(data_dir, video1)
        video_path2 = os.path.join(data_dir, video2)
        gpt_result = self.compare_two_videos_with_gpt(video_path1, video_path2, property_type, frame_size=frame_size, model_version=model_version, concat_mode=concat_mode, frame_index=frame_index, icl_examples=icl_examples, follow_oracle_test=follow_oracle_test, max_retries=max_retries, retry_delay=retry_delay, validation_retries=validation_retries)
        predicted_comparison = self.extract_comparison_result(gpt_result)
        print(f'Success: predicted={predicted_comparison}, gt={gt_comparison}')
        return {'video_pair': f'{video1} vs {video2}', 'predicted': predicted_comparison, 'gt': gt_comparison, 'gt_values': f'{gt_value1} vs {gt_value2}', 'gpt_response': gpt_result}

    def get_gt_value(self, video_file: str, gt_data: Optional[Dict], dataset_class: str, property_type: str) -> float:
        basename = video_file.replace('_rgb.mp4', '')
        if gt_data and basename in gt_data:
            gt_value = gt_data[basename]
            if isinstance(gt_value, dict):
                if property_type == 'elasticity':
                    if 'gt_restitution' in gt_value:
                        return gt_value['gt_restitution']
                    elif 'sqrt_ratio' in gt_value:
                        return gt_value['sqrt_ratio']
                elif property_type in ['friction', 'viscosity']:
                    if 'gt_restitution' in gt_value:
                        return gt_value['gt_restitution']
                for key, value in gt_value.items():
                    if isinstance(value, (int, float)):
                        return float(value)
                print(f'Warning: GT data format not recognized:{gt_value}')
                return 0.0
            else:
                return float(gt_value)
        else:
            parsed_value = self.extract_property_from_filename(video_file, property_type, dataset_class)
            if parsed_value != 0.0:
                return parsed_value
            print(f'Warning: Video cannot be found{video_file}GT value (basename:{basename})')
            if gt_data:
                print(f'Example of keys in GT file:{list(gt_data.keys())[:5]}...')
            return 0.0

    def extract_predicted_value(self, gpt_response: str, property_type: str) -> Optional[float]:
        print(f'=== GPT complete reply ({property_type}) ===')
        print(gpt_response)
        print('=== End of reply ===')
        if not gpt_response or len(gpt_response.strip()) == 0:
            print('✗ GPT returns empty reply')
            return None
        error_keywords = ['API call failed', 'call failed', 'Server is busy', 'Please try again later', 'All retries failed', 'GPT API call error', 'GPT Vision API call failed', 'chatanywhere_error', 'INTERNAL_SERVER_ERROR', 'error', 'Third-party API call failed']
        for keyword in error_keywords:
            if keyword in gpt_response:
                print(f'✗ API error message detected, including keywords:{keyword}')
                return None
        try:
            if property_type == 'friction':
                pattern = 'Estimated friction coefficient value:\\s*(\\d+\\.?\\d*)'
                match = re.search(pattern, gpt_response, re.IGNORECASE)
                if match:
                    result = float(match.group(1))
                    print(f'✓ Find friction format matching:{result}')
                    return result
            elif property_type == 'viscosity':
                patterns = ['Estimated viscosity value:\\s*(\\d+\\.?\\d*)\\s*cP', 'Estimated viscosity value:\\s*(\\d+\\.?\\d*)', 'viscosity.*?value.*?(\\d+\\.?\\d*)\\s*cP', 'viscosity.*?estimate.*?(\\d+\\.?\\d*)\\s*cP']
                for pattern in patterns:
                    match = re.search(pattern, gpt_response, re.IGNORECASE)
                    if match:
                        result = float(match.group(1))
                        print(f'✓ Find viscosity format matching:{result}')
                        return result
            elif property_type == 'elasticity':
                patterns = ['Estimated restitution coefficient value:\\s*(\\d+\\.?\\d*)', 'restitution.*?coefficient.*?value.*?(\\d+\\.?\\d*)', 'restitution.*?coefficient.*?is.*?(\\d+\\.?\\d*)', 'coefficient.*?of.*?restitution.*?(\\d+\\.?\\d*)', 'elasticity.*?value.*?(\\d+\\.?\\d*)', 'elasticity.*?coefficient.*?(\\d+\\.?\\d*)']
                for pattern in patterns:
                    match = re.search(pattern, gpt_response, re.IGNORECASE)
                    if match:
                        result = float(match.group(1))
                        if 0 <= result <= 1:
                            print(f'✓ Find elasticity format matching:{result}')
                            return result
                        else:
                            print(f'⚠️The elasticity value is out of the range [0,1]:{result}')
                            continue
            print('⚠️ No standard format found, try smart extraction...')
            lines = gpt_response.split('\n')
            for line in lines:
                if any((keyword in line.lower() for keyword in error_keywords)):
                    continue
                if property_type == 'elasticity':
                    if any((word in line.lower() for word in ['coefficient', 'restitution', 'elasticity', 'bounce', 'value', 'estimate'])):
                        numbers = re.findall('\\d+\\.?\\d*', line)
                        for num_str in numbers:
                            try:
                                num = float(num_str)
                                if 0 <= num <= 1:
                                    print(f'⚠️ Extract elasticity values ​​from relevant rows:{num}(OK:{line.strip()})')
                                    return num
                            except ValueError:
                                continue
                elif 'estimate' in line.lower() or 'value' in line.lower():
                    numbers = re.findall('\\d+\\.?\\d*', line)
                    if numbers:
                        result = float(numbers[-1])
                        print(f'⚠️ Extract from relevant rows:{result}(OK:{line.strip()})')
                        return result
            numbers = re.findall('\\d+\\.?\\d*', gpt_response)
            if numbers:
                filtered_numbers = []
                if property_type == 'elasticity':
                    for n in numbers:
                        try:
                            val = float(n)
                            if 0 <= val <= 1:
                                filtered_numbers.append(n)
                        except ValueError:
                            continue
                else:
                    reference_values = {'1', '100', '300', '5000', '0.02', '0.03', '0.4', '0.6', '0.8', '0.3', '0.5', '500', '400', '404', '502', '503', '504', '200', '201', '10001'}
                    filtered_numbers = [n for n in numbers if n not in reference_values]
                if filtered_numbers:
                    result = float(filtered_numbers[-1])
                    print(f'⚠️ Extract after filtering:{result}')
                    return result
                else:
                    print('⚠️ All numbers are filtered, most likely an error response')
                    return None
        except (ValueError, AttributeError) as e:
            print(f'✗ Failed to extract:{e}')
            pass
        print('✗ Unable to extract predicted values')
        return None

    def extract_comparison_result(self, gpt_response: str) -> Optional[float]:
        print(f'=== GPT complete reply (relative comparison) ===')
        print(gpt_response)
        print('=== End of reply ===')
        if not gpt_response or len(gpt_response.strip()) == 0:
            print('✗ GPT returns empty reply')
            return None
        error_keywords = ['API call failed', 'call failed', 'Server is busy', 'Please try again later', 'All retries failed', 'GPT API call error', 'GPT Vision API call failed', 'chatanywhere_error', 'INTERNAL_SERVER_ERROR', 'error', 'Third-party API call failed']
        for keyword in error_keywords:
            if keyword in gpt_response:
                print(f'✗ API error message detected, including keywords:{keyword}')
                return None
        try:
            pattern = 'Comparison result:\\s*([0-1](?:\\.\\d+)?)'
            match = re.search(pattern, gpt_response, re.IGNORECASE)
            if match:
                result = float(match.group(1))
                if 0.0 <= result <= 1.0:
                    print(f'✓ Find standard confidence format matches:{result}')
                    return result
                else:
                    print(f'⚠️ Confidence value is out of range [0,1]:{result}')
            decimal_pattern = '([0-1](?:\\.\\d+))'
            decimals = re.findall(decimal_pattern, gpt_response)
            if decimals:
                for decimal in reversed(decimals):
                    try:
                        result = float(decimal)
                        if 0.0 <= result <= 1.0:
                            print(f'⚠️ Extract confidence value from text:{result}')
                            return result
                    except ValueError:
                        continue
            lower_response = gpt_response.lower()
            video1_patterns = ['video 1.*?(?:has|shows?).*?(?:higher|more|greater)', 'video 1.*?(?:higher|more|greater)', '(?:higher|more|greater).*?video 1']
            video2_patterns = ['video 2.*?(?:has|shows?).*?(?:higher|more|greater)', 'video 2.*?(?:higher|more|greater)', '(?:higher|more|greater).*?video 2']
            video1_matches = 0
            video2_matches = 0
            for pattern in video1_patterns:
                if re.search(pattern, lower_response):
                    video1_matches += 1
            for pattern in video2_patterns:
                if re.search(pattern, lower_response):
                    video2_matches += 1
            if video1_matches > video2_matches:
                print('⚠️ Inference from text: bias towards VIDEO 1, set confidence level 0.8')
                return 0.8
            elif video2_matches > video1_matches:
                print('⚠️ Infer from text: bias towards VIDEO 2, set confidence level to 0.2')
                return 0.2
            equal_keywords = ['equal', 'same', 'similar', 'uncertain', 'unclear', 'difficult to determine']
            for keyword in equal_keywords:
                if keyword in lower_response:
                    print(f"⚠️ Uncertainty keyword detected'{keyword}', set the confidence level to 0.5")
                    return 0.5
        except (ValueError, AttributeError) as e:
            print(f'✗ Confidence extraction failed:{e}')
            pass
        print('✗ Unable to extract confidence score')
        return None

def main():
    print('=' * 80)
    print('GPT Physics Analyzer - Comprehensive physical property analysis tool')
    print('=' * 80)
    try:
        analyzer = GPTPhysicsAnalyzer()
        print('✓ GPT Physics Analyzer initialization completed')
        print(f"✓ Supported attributes:{', '.join(analyzer.properties)}")
        print(f"✓ Supported analysis types:{', '.join(analyzer.analysis_types)}")
        print(f"✓ Configured dataset versions: {len(analyzer.config)}")
        print('\nConfigured versions:')
        for version, config in analyzer.config.items():
            print(f"  - {version}: {config['property']} ({config['type']})")
            for test_id in config['data_dirs'].keys():
                print(f'    └── {test_id}')
        print('\n' + '=' * 80)
        response = input('Start running all tests? (y/n):').lower().strip()
        if response in ['y', 'yes', 'yes']:
            print('\nStart running tests...')
            output_dir = 'gpt_physics_analysis_results'
            frame_size = (640, 480)
            results = analyzer.run_all_tests(output_dir=output_dir, frame_size=frame_size)
            print('\n' + '=' * 80)
            print('Test completion summary:')
            total_tests = len(results)
            successful_tests = sum((1 for r in results if r['successful_analyses'] > 0))
            print(f'Total number of tests:{total_tests}')
            print(f'Number of successful tests:{successful_tests}')
            print(f'Number of failed tests:{total_tests - successful_tests}')
            for property_type in analyzer.properties:
                property_results = [r for r in results if r['property'] == property_type]
                if property_results:
                    print(f'\n{property_type.upper()}Test results:')
                    for result in property_results:
                        status = '✓' if result['successful_analyses'] > 0 else '✗'
                        print(f"  {status} {result['version']} ({result['type']}) - {result['test_set']}: {result['successful_analyses']}/{result['total_samples']}")
        else:
            print('Cancel test run')
    except Exception as e:
        print(f'✗ Initialization failed:{str(e)}')
        return 1
    return 0
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    exit_code = main()
    sys.exit(exit_code)
