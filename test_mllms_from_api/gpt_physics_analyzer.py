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
    """
    综合GPT物理属性分析器
    支持friction、viscosity、elasticity的单值预测和相对比较
    """
    
    def __init__(self, config_path: str = "vjepa_data_dirs.json"):
        """
        初始化分析器
        
        参数:
            config_path: vjepa数据目录配置文件路径
        """
        # 如果config_path是相对路径，则相对于当前脚本目录
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        
        self.config_path = config_path
        self.load_config()
        
        # 支持的物理属性和分析类型
        self.properties = ["friction", "viscosity", "elasticity"]
        self.analysis_types = ["absolute", "relative"]
        
        # 数据集类与GT文件的映射关系
        self.dataset_gt_mapping = {
            "WebVidv2Viscosity": "bounce_analysis_results.json",      # v119 viscosity absolute, v120 friction absolute, v124 elasticity absolute
            "WebVidv3ViscosityTest": "bounce_analysis_results.json",  # v110 viscosity relative - 实际上有GT文件！
            "WebVidv3FrictionTest": "bounce_analysis_results.json",   # v121 friction relative - 实际上有GT文件！  
            "WebVidv3ElasticityTest": "gt_retitution1.json"           # v125 elasticity relative
        }
        
        # 不同版本对应的数据集类
        self.version_dataset_mapping = {
            "v119_vjepa": "WebVidv2Viscosity",      # viscosity absolute
            "v110_vjepa": "WebVidv3ViscosityTest",  # viscosity relative
            "v120_vjepa": "WebVidv2Viscosity",      # friction absolute  
            "v121_vjepa": "WebVidv3FrictionTest",   # friction relative
            "v124_vjepa": "WebVidv2Viscosity",      # elasticity absolute (应该使用gt_retitution1.json)
            "v125_vjepa": "WebVidv3ElasticityTest"  # elasticity relative
        }
        
        # 特殊情况：v124需要使用gt_retitution1.json而不是bounce_analysis_results.json
        self.version_gt_override = {
            "v124_vjepa": "gt_retitution1.json"
        }
    
    def load_config(self):
        """加载vjepa数据目录配置"""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        print(f"已加载配置: {len(self.config)} 个测试版本")
    
    def get_gt_data(self, data_dir: str, dataset_class: str, property_type: str, version: str = None) -> Optional[Dict]:
        """
        根据数据集类型加载对应的GT数据
        
        参数:
            data_dir: 数据目录路径
            dataset_class: 数据集类名
            property_type: 物理属性类型
            version: 版本名（用于特殊情况处理）
            
        返回:
            GT数据字典或None
        """
        # 检查是否有版本特殊覆盖
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
                print(f"警告: GT文件不存在 - {gt_path}")
                return None
        else:
            # 对于WebVidv3系列，从文件名解析属性值
            print(f"使用文件名解析GT数据，数据集类: {dataset_class}")
            return {}
    
    def extract_property_from_filename(self, filename: str, property_type: str, dataset_class: str) -> float:
        """
        从文件名提取物理属性值
        
        参数:
            filename: 视频文件名
            property_type: 物理属性类型
            dataset_class: 数据集类名
            
        返回:
            提取的属性值
        """
        basename = filename.replace('_rgb.mp4', '')
        parts = basename.split('_')
        
        try:
            # 对于WebVidv3系列，统一从objFriction后面的位置提取
            if dataset_class in ["WebVidv3ViscosityTest", "WebVidv3FrictionTest"]:
                # 找到"objFriction"的位置
                if "objFriction" in parts:
                    obj_friction_index = parts.index("objFriction")
                    if obj_friction_index + 1 < len(parts):
                        return float(parts[obj_friction_index + 1])
                
                # 如果找不到objFriction，尝试其他可能的位置
                for i, part in enumerate(parts):
                    # 尝试解析可能的数值
                    try:
                        val = float(part)
                        # 过滤掉明显不是属性值的数字（如id、版本号等）
                        if val > 0 and val not in [0, 1, 2] and '.' in part:
                            return val
                    except ValueError:
                        continue
                        
                return 0.0
            
            # 对于WebVidv2系列的friction测试，也从objFriction后面提取
            elif dataset_class == "WebVidv2Viscosity" and property_type == "friction":
                # 找到"objFriction"的位置
                if "objFriction" in parts:
                    obj_friction_index = parts.index("objFriction")
                    if obj_friction_index + 1 < len(parts):
                        return float(parts[obj_friction_index + 1])
                        
                return 0.0
            else:
                return 0.0
        except (IndexError, ValueError):
            return 0.0
    
    def sample_icl_examples(self, train_dir: str, property_type: str, analysis_type: str, dataset_class: str, num_examples: int = None, frame_size: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        从训练集中采样ICL示例
        
        参数:
            train_dir: 训练集目录路径
            property_type: 物理属性类型
            analysis_type: 分析类型 (absolute/relative)
            dataset_class: 数据集类名
            num_examples: 示例数量 (相对比较测试建议使用1-2个示例以避免API限制)
            frame_size: 帧大小
            
        返回:
            ICL示例列表，每个示例包含input和output
        """
        if not os.path.exists(train_dir):
            print(f"警告: 训练集目录不存在 - {train_dir}")
            return []
        
        # 加载训练集GT数据
        gt_data = self.get_gt_data(train_dir, dataset_class, property_type)
        
        if not gt_data:
            print(f"警告: 无法加载GT数据 - {train_dir}")
            return []
        
        # 只从GT文件中存在的键对应的视频文件中采样
        available_videos = []
        for basename in gt_data.keys():
            video_file = basename + '_rgb.mp4'
            video_path = os.path.join(train_dir, video_file)
            if os.path.exists(video_path):
                available_videos.append(video_file)
        
        available_videos.sort()
        print(f"从GT数据中找到 {len(available_videos)} 个可用的视频文件（总GT键数: {len(gt_data)}）")
        
        # 设置默认示例数量
        if num_examples is None:
            if analysis_type == "relative":
                num_examples = 1  # 相对比较测试默认使用1个示例
            else:
                num_examples = 1  # 绝对值测试默认使用1个示例（避免API限制）
        
        icl_examples = []
        
        if analysis_type == "absolute":
            # 绝对值测试：从有效视频中采样
            if len(available_videos) == 0:
                print("警告: 没有找到有效的视频文件用于ICL采样")
                return []
            
            # 确定性采样：均匀分布选择
            sample_size = min(len(available_videos), num_examples)
            if sample_size == len(available_videos):
                selected_videos = available_videos
            else:
                step = len(available_videos) / sample_size
                indices = [int(i * step) for i in range(sample_size)]
                selected_videos = [available_videos[i] for i in indices]
            
            for video_file in selected_videos:
                try:
                    # 获取GT值（由于是从GT数据中选择的，应该都有有效值）
                    gt_value = self.get_gt_value(video_file, gt_data, dataset_class, property_type)
                    
                    # 双重检查GT值有效性
                    if gt_value == 0.0:
                        print(f"意外：GT值为0的视频: {video_file} (GT=0.0)")
                        continue
                    
                    # 读取视频帧
                    video_path = os.path.join(train_dir, video_file)
                    frames = self.read_video_frames(video_path, target_frames=16, frame_size=frame_size)
                    
                    # 构建绝对值测试的输出格式
                    if property_type == "friction":
                        output = f"Estimated friction coefficient value: {gt_value:.3f}"
                    elif property_type == "viscosity":
                        output = f"Estimated viscosity value: {gt_value:.1f} cP"
                    elif property_type == "elasticity":
                        output = f"Estimated restitution coefficient value: {gt_value:.3f}"
                    else:
                        output = f"Estimated {property_type} value: {gt_value:.3f}"
                    
                    icl_examples.append({
                        'video_file': video_file,
                        'frames': frames,
                        'gt_value': gt_value,
                        'output': output,
                        'type': 'absolute'
                    })
                    
                    print(f"成功采样ICL示例: {video_file} (GT={gt_value:.3f})")
                    
                except Exception as e:
                    print(f"警告: 绝对值ICL示例采样失败 - {video_file}: {str(e)}")
                    continue
            
            if len(icl_examples) < num_examples:
                print(f"警告: 只找到 {len(icl_examples)}/{num_examples} 个有效的ICL示例")
        
        else:  # relative comparison
            # 相对比较测试：采样成对视频
            print(f"开始为相对比较测试采样ICL示例...")
            
            # 对于相对比较测试，建议使用较少的示例以避免API限制
            if num_examples > 2:
                print(f"警告: 相对比较测试建议使用1-2个示例以避免API限制，当前设置为{num_examples}")
                print(f"自动调整为1个示例...")
                num_examples = 1
            
            # 首先按video_x分组，寻找成对的视频
            video_groups = {}
            for video_file in available_videos:
                if 'video_' in video_file and '_id_' in video_file:
                    parts = video_file.split('_')
                    
                    # 查找video_x_id_y的模式
                    for i, part in enumerate(parts):
                        if part == 'video' and i + 2 < len(parts) and parts[i + 2] == 'id':
                            video_num = f"video_{parts[i + 1]}"  # video_x
                            id_part = parts[i + 3]  # id_y
                            
                            if video_num not in video_groups:
                                video_groups[video_num] = []
                            video_groups[video_num].append((video_file, id_part))
                            break
            
            print(f"找到 {len(video_groups)} 个video组用于ICL采样")
            
            # 从video组中选择成对的视频作为ICL示例
            sampled_pairs = 0
            video_group_list = list(video_groups.items())
            
            # 确保至少有足够的组来生成示例
            if len(video_group_list) < num_examples:
                print(f"警告: video组数量不足 - {len(video_group_list)} < {num_examples}")
                num_examples = len(video_group_list)
            
            # 确定性选择组（均匀分布）
            if num_examples == len(video_group_list):
                selected_groups = video_group_list
            else:
                step = len(video_group_list) / num_examples
                indices = [int(i * step) for i in range(num_examples)]
                selected_groups = [video_group_list[i] for i in indices]
            
            for video_num, videos in selected_groups:
                if sampled_pairs >= num_examples:
                    break
                    
                # 按id排序
                videos.sort(key=lambda x: x[1])
                
                # 寻找id_0和id_1作为一对（如果没有则选择前两个）
                if len(videos) >= 2:
                    video_pair = videos[:2]  # 选择前两个视频
                    video_file1, id1 = video_pair[0]
                    video_file2, id2 = video_pair[1]
                    
                    try:
                        # 获取两个视频的GT值，先检查有效性
                        gt_value1 = self.get_gt_value(video_file1, gt_data, dataset_class, property_type)
                        gt_value2 = self.get_gt_value(video_file2, gt_data, dataset_class, property_type)
                        
                        # 跳过GT值无效的视频对
                        if gt_value1 == 0.0 or gt_value2 == 0.0:
                            print(f"跳过无效GT值的视频对: {video_file1} (GT={gt_value1:.3f}) vs {video_file2} (GT={gt_value2:.3f})")
                            continue
                        
                        # 读取两个视频的帧
                        video_path1 = os.path.join(train_dir, video_file1)
                        video_path2 = os.path.join(train_dir, video_file2)
                        frames1 = self.read_video_frames(video_path1, target_frames=16, frame_size=frame_size)
                        frames2 = self.read_video_frames(video_path2, target_frames=16, frame_size=frame_size)
                        
                        # 基于真实GT值计算比较结果（使用高置信度）
                        if gt_value1 > gt_value2:
                            gt_comparison = 1
                            confidence_score = 0.9  # 高置信度偏向VIDEO 1
                        elif gt_value2 > gt_value1:
                            gt_comparison = 0
                            confidence_score = 0.1  # 高置信度偏向VIDEO 2
                        else:
                            gt_comparison = 0.5
                            confidence_score = 0.5  # 相等情况
                        
                        # 构建相对比较的置信度输出格式
                        output = f"Comparison result: {confidence_score}"
                        
                        icl_examples.append({
                            'video_file1': video_file1,
                            'video_file2': video_file2,
                            'frames1': frames1,
                            'frames2': frames2,
                            'gt_value1': gt_value1,
                            'gt_value2': gt_value2,
                            'gt_comparison': gt_comparison,
                            'output': output,
                            'type': 'relative'
                        })
                        
                        sampled_pairs += 1
                        print(f"成功采样相对比较ICL示例 {sampled_pairs}: {video_file1} vs {video_file2} (GT: {gt_value1:.3f} vs {gt_value2:.3f} → {gt_comparison}, 置信度: {confidence_score})")
                        
                    except Exception as e:
                        print(f"警告: 相对比较ICL示例采样失败 - {video_file1} vs {video_file2}: {str(e)}")
                        continue
                else:
                    print(f"警告: 组 {video_num} 视频数量不足以形成配对")
            
            # 如果没有成功的video组配对，尝试直接选择视频配对
            if len(icl_examples) == 0 and len(available_videos) >= 2:
                print("尝试直接从有效视频中配对...")
                num_pairs = min(num_examples, len(available_videos) // 2)
                
                # 从有效视频中直接配对
                for i in range(num_pairs):
                    if len(icl_examples) >= num_examples:
                        break
                        
                    video_file1 = available_videos[i * 2]
                    video_file2 = available_videos[i * 2 + 1]
                    
                    try:
                        # 获取两个视频的GT值（应该都有效）
                        gt_value1 = self.get_gt_value(video_file1, gt_data, dataset_class, property_type)
                        gt_value2 = self.get_gt_value(video_file2, gt_data, dataset_class, property_type)
                        
                        # 双重检查GT值有效性
                        if gt_value1 == 0.0 or gt_value2 == 0.0:
                            print(f"意外：GT值无效的视频对: {video_file1} (GT={gt_value1:.3f}) vs {video_file2} (GT={gt_value2:.3f})")
                            continue
                        
                        # 读取两个视频的帧
                        video_path1 = os.path.join(train_dir, video_file1)
                        video_path2 = os.path.join(train_dir, video_file2)
                        frames1 = self.read_video_frames(video_path1, target_frames=16, frame_size=frame_size)
                        frames2 = self.read_video_frames(video_path2, target_frames=16, frame_size=frame_size)
                        
                        # 基于真实GT值计算比较结果（使用高置信度）
                        if gt_value1 > gt_value2:
                            gt_comparison = 1
                            confidence_score = 0.9  # 高置信度偏向VIDEO 1
                        elif gt_value2 > gt_value1:
                            gt_comparison = 0
                            confidence_score = 0.1  # 高置信度偏向VIDEO 2
                        else:
                            gt_comparison = 0.5
                            confidence_score = 0.5  # 相等情况
                        
                        # 构建相对比较的置信度输出格式
                        output = f"Comparison result: {confidence_score}"
                        
                        icl_examples.append({
                            'video_file1': video_file1,
                            'video_file2': video_file2,
                            'frames1': frames1,
                            'frames2': frames2,
                            'gt_value1': gt_value1,
                            'gt_value2': gt_value2,
                            'gt_comparison': gt_comparison,
                            'output': output,
                            'type': 'relative'
                        })
                        
                        print(f"成功采样直接配对ICL示例 {i+1}: {video_file1} vs {video_file2} (GT: {gt_value1:.3f} vs {gt_value2:.3f} → {gt_comparison}, 置信度: {confidence_score})")
                        
                    except Exception as e:
                        print(f"警告: 直接配对ICL示例采样失败 - {video_file1} vs {video_file2}: {str(e)}")
                        continue
        
        print(f"成功采样 {len(icl_examples)} 个ICL示例 (类型: {analysis_type})")
        return icl_examples

    def sample_videos(self, data_dir: str, max_samples: int = 50) -> List[str]:
        """
        从数据目录确定性采样视频文件（按文件名排序后均匀采样）
        
        参数:
            data_dir: 数据目录路径
            max_samples: 最大采样数量
            
        返回:
            采样的视频文件名列表
        """
        if not os.path.exists(data_dir):
            print(f"警告: 数据目录不存在 - {data_dir}")
            return []
        
        # 获取所有rgb.mp4文件并按文件名排序
        all_videos = [f for f in os.listdir(data_dir) if f.endswith('_rgb.mp4')]
        all_videos.sort()  # 按文件名排序，确保每次运行结果一致
        
        if len(all_videos) == 0:
            print(f"警告: 目录中未找到rgb.mp4文件 - {data_dir}")
            return []
        
        # 确定性采样：均匀分布选择
        sample_size = min(len(all_videos), max_samples)
        if sample_size == len(all_videos):
            # 如果样本数不够，全部返回
            sampled_videos = all_videos
        else:
            # 均匀采样
            step = len(all_videos) / sample_size
            indices = [int(i * step) for i in range(sample_size)]
            sampled_videos = [all_videos[i] for i in indices]
        
        print(f"从 {data_dir} 中确定性采样了 {sample_size}/{len(all_videos)} 个视频")
        return sampled_videos
    
    def read_video_frames(self, video_path: str, target_frames: int = 16, frame_size: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        读取视频文件并将帧转换为base64编码，智能采样以达到目标帧数
        
        参数:
            video_path: 视频文件路径
            target_frames: 目标帧数
            frame_size: 调整帧大小的元组 (宽度, 高度)
            
        返回:
            base64编码的视频帧列表
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 先读取所有帧
        all_frames = []
        frame_count = 0
        
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
                
            # 如果指定了帧大小，则调整帧的大小
            if frame_size is not None:
                frame = cv2.resize(frame, frame_size)
                
            _, buffer = cv2.imencode(".jpg", frame)
            all_frames.append(base64.b64encode(buffer).decode("utf-8"))
            frame_count += 1

        video.release()
        
        # 智能采样策略
        if frame_count <= target_frames:
            # 如果视频帧数不够，全部保留
            selected_frames = all_frames
            print(f"视频 {os.path.basename(video_path)}: 总帧数 {frame_count}，全部保留 {len(selected_frames)} 帧")
        else:
            # 如果视频帧数够，均匀采样
            indices = [int(i * (frame_count - 1) / (target_frames - 1)) for i in range(target_frames)]
            selected_frames = [all_frames[i] for i in indices]
            print(f"视频 {os.path.basename(video_path)}: 总帧数 {frame_count}，均匀采样 {len(selected_frames)} 帧")
        
        return selected_frames
    
    def build_icl_content(self, icl_examples: List[Dict], frame_index: bool = False, concat_mode: bool = False) -> List[Dict]:
        """
        构建ICL示例的content部分
        
        参数:
            icl_examples: ICL示例列表
            frame_index: 是否启用帧索引
            concat_mode: 是否使用拼接模式
            
        返回:
            ICL content列表
        """
        icl_content = []
        
        for i, example in enumerate(icl_examples, 1):
            # 添加示例标题
            icl_content.append({"type": "text", "text": f"Example {i}:"})
            
            if example.get('type') == 'absolute':
                # 绝对值测试：添加单个视频的帧
                if frame_index:
                    for j, frame in enumerate(example['frames'], 1):
                        icl_content.append({"type": "text", "text": f"frame{j}:"})
                        icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                else:
                    for frame in example['frames']:
                        icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
            
            elif example.get('type') == 'relative':
                # 相对比较测试：添加两个视频的帧
                if concat_mode:
                    # 拼接模式：video1 + 黑帧 + video2，无额外标签
                    # 生成黑帧（使用默认尺寸）
                    black_frames = self.generate_black_frames(num_frames=3, frame_size=(640, 480))
                    
                    if frame_index:
                        # 拼接模式 + 帧索引：按顺序添加所有帧
                        frame_count = 1
                        
                        # 添加video1帧
                        for frame in example['frames1']:
                            icl_content.append({"type": "text", "text": f"frame{frame_count}:"})
                            icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                            frame_count += 1
                        
                        # 添加黑帧（不计入帧索引）
                        for black_frame in black_frames:
                            icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{black_frame}"}})
                        
                        # 添加video2帧
                        for frame in example['frames2']:
                            icl_content.append({"type": "text", "text": f"frame{frame_count}:"})
                            icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                            frame_count += 1
                    else:
                        # 拼接模式，无帧索引：直接拼接
                        for frame in example['frames1']:
                            icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                        
                        for black_frame in black_frames:
                            icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{black_frame}"}})
                        
                        for frame in example['frames2']:
                            icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                else:
                    # 传统模式：分别标识两个视频
                    icl_content.append({"type": "text", "text": "VIDEO 1 frames:"})
                    
                    if frame_index:
                        for j, frame in enumerate(example['frames1'], 1):
                            icl_content.append({"type": "text", "text": f"frame{j}:"})
                            icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                    else:
                        for frame in example['frames1']:
                            icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                    
                    icl_content.append({"type": "text", "text": "VIDEO 2 frames:"})
                    
                    if frame_index:
                        for j, frame in enumerate(example['frames2'], 1):
                            icl_content.append({"type": "text", "text": f"frame{j}:"})
                            icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                    else:
                        for frame in example['frames2']:
                            icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
            
            # 兼容旧格式（没有type字段）- 需要判断是绝对值还是相对比较测试
            if 'frames1' in example and 'frames2' in example:
                # 相对比较测试的ICL示例
                icl_content.append({"type": "text", "text": "VIDEO 1 frames:"})
                
                if frame_index:
                    for j, frame in enumerate(example['frames1'], 1):
                        icl_content.append({"type": "text", "text": f"frame{j}:"})
                        icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                else:
                    for frame in example['frames1']:
                        icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                
                icl_content.append({"type": "text", "text": "VIDEO 2 frames:"})
                
                if frame_index:
                    for j, frame in enumerate(example['frames2'], 1):
                        icl_content.append({"type": "text", "text": f"frame{j}:"})
                        icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                else:
                    for frame in example['frames2']:
                        icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
            else:
                # 绝对值测试的ICL示例
                if frame_index:
                    for j, frame in enumerate(example["frames"], 1):
                        icl_content.append({"type": "text", "text": f"frame{j}:"})
                        icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                else:
                    for frame in example['frames']:
                        icl_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
            
            # 添加输出示例
            icl_content.append({"type": "text", "text": f"Output: {example['output']}"})
            icl_content.append({"type": "text", "text": ""})  # 空行分隔
        
        # 添加当前任务标识
        icl_content.append({"type": "text", "text": "Now analyze the following video:"})
        
        return icl_content

    def get_analysis_prompt(self, property_type: str, analysis_type: str, icl_examples: Optional[List[Dict]] = None, follow_oracle_test: bool = False) -> str:
        """
        获取分析提示词
        
        参数:
            property_type: 物理属性类型 (friction/viscosity/elasticity)
            analysis_type: 分析类型 (absolute/relative)
            icl_examples: ICL示例列表（如果提供，会修改提示词以适应ICL格式）
            follow_oracle_test: 是否包含解题步骤指导
            
        返回:
            提示词字符串
        """
        # 根据是否有ICL示例调整提示词
        icl_prefix = ""
        if icl_examples:
            icl_prefix = "Based on the examples above, "
        
        # 定义解题步骤指导内容
        oracle_guidance = ""
        if follow_oracle_test:
            oracle_guidance_dict = {
                "elasticity": """
ANALYSIS METHOD:
The intrinsic visual cue to reflect the elasticity property is the dropping height and bouncing height of the ball. You only need to calculate the ratio of the height difference in the camera coordinate. From this trajectory, you need to identify key points: the initial position, first ground contact, and bounce peak.

Follow these steps:
1. Identify the ball's initial dropping position (highest point before impact)
2. Locate the first ground contact point (lowest point during impact)
3. Find the bounce peak (highest point after first bounce)
4. Calculate the height ratio: (bounce height - ground level) / (initial height - ground level)
5. This ratio represents the restitution coefficient (elasticity)

""",
                "viscosity": """
ANALYSIS METHOD:
The intrinsic visual cue to reflect the viscosity property is the speed that the liquid expands on the ground. You can calculate the area size in the camera plane and then normalise the area sizes by the first area size when the liquid touches ground; the growth speed of the normalised area size sequence reflects the viscosity.

Follow these steps:
1. Identify when the liquid first touches the ground (initial contact frame)
2. Measure the liquid area size in each subsequent frame
3. Normalize all area sizes by dividing by the initial contact area
4. Calculate the growth rate of the normalized area sequence over time
5. Higher viscosity = slower area expansion, lower viscosity = faster area expansion

""",
                "friction": """
ANALYSIS METHOD:
The intrinsic visual cue to reflect the friction property is the acceleration of the sliding object, i.e., how fast is the velocity of the object reduced because of friction and can be inferred by the position of the object over time. You can estimate a homography transformation from the 4 corner points of the top surface of the object. You then use the homography transformation to project the object upper plane to bird eye view, and estimate the acceleration a via fitting a parabola ax² + bx + c to the position of the object over time.

Follow these steps:
1. Identify the 4 corner points of the object's top surface in each frame
2. Estimate homography transformation to convert to bird's eye view
3. Track the object's position over time in the transformed coordinate system
4. Fit a parabolic function (ax² + bx + c) to the position-time data
5. Extract the acceleration coefficient 'a' which reflects the friction level
6. Higher friction = larger deceleration (more negative 'a'), lower friction = smaller deceleration

"""
            }
            oracle_guidance = oracle_guidance_dict.get(property_type, "")
        
        if analysis_type == "absolute":
            if property_type == "friction":
                return f"""
                You are analyzing frames from a video showing object motion with friction. Your task is to estimate the coefficient of friction.
                
                IMPORTANT: You CAN analyze these images. You MUST provide a specific friction coefficient estimate.
                
                {oracle_guidance}The coefficient of friction is typically between 0 and 1:
                - Very smooth surfaces (ice on ice): 0.02-0.03
                - Steel on steel: 0.4-0.6  
                - Rubber on concrete: 0.6-0.8
                - Wood on wood: 0.3-0.5
                
                {icl_prefix}analyze the object's deceleration, sliding behavior, and surface interaction.
                
                YOUR RESPONSE MUST END WITH A SPECIFIC FRICTION COEFFICIENT VALUE IN THIS FORMAT:
                "Estimated friction coefficient value: X"
                
                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.
                """
                
            elif property_type == "viscosity":
                return f"""
                You are analyzing frames from a video showing liquid motion. Your task is to estimate the viscosity of this liquid.
                
                IMPORTANT: You CAN analyze these images. You MUST provide a specific viscosity estimate.
                
                {oracle_guidance}For reference:
                - Water: 1 cP
                - Vegetable oil: 100 cP
                - Motor oil: 300 cP
                - Honey: 5000 cP
                
                {icl_prefix}analyze the liquid's flow speed, surface waves, and interaction with objects.
                
                YOUR RESPONSE MUST END WITH A SPECIFIC VISCOSITY VALUE IN THIS FORMAT:
                "Estimated viscosity value: X cP"
                
                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.
                """
                
            elif property_type == "elasticity":
                return f"""
                You are analyzing frames from a video showing a bouncing object. Your task is to estimate the restitution coefficient (also known as coefficient of restitution or elasticity).
                
                IMPORTANT: You CAN analyze these images. You MUST provide a specific restitution coefficient estimate.
                
                {oracle_guidance}The restitution coefficient is the ratio of separation speed to approach speed during collision:
                - Perfectly elastic collision: 1.0 (ball bounces back with same speed)
                - Perfectly inelastic collision: 0.0 (ball sticks to surface, no bounce)
                - Typical values: 0.1-0.9 for most materials
                
                {icl_prefix}analyze the object's behavior:
                1. Look for the object falling towards a surface
                2. Observe the impact moment
                3. Check if and how the object bounces back
                4. Compare the approach speed vs. separation speed
                
                If the object bounces back significantly → restitution coefficient closer to 1.0
                If the object barely bounces or sticks → restitution coefficient closer to 0.0
                
                YOUR RESPONSE MUST END WITH A SPECIFIC RESTITUTION COEFFICIENT VALUE IN THIS FORMAT:
                "Estimated restitution coefficient value: X"
                
                Where X is a number between 0 and 1 (e.g., 0.7, 0.3, 0.85).
                
                Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.
                """
        
        elif analysis_type == "relative":  # relative comparison (original mode)
            return f"""
            You are comparing two {property_type} videos to determine which has a higher {property_type} value.
            
            IMPORTANT: You CAN analyze these images. You MUST provide a confidence score between 0 and 1.
            
            {oracle_guidance}The first set of frames is labeled "VIDEO 1" and the second set is labeled "VIDEO 2".
            
            {icl_prefix}compare the {property_type} characteristics between the two videos:
            - VIDEO 1 frames are shown first
            - VIDEO 2 frames are shown second
            
            YOUR RESPONSE MUST END WITH A SPECIFIC CONFIDENCE SCORE IN THIS FORMAT:
            "Comparison result: X"
            
            Where X is a decimal number between 0.0 and 1.0:
            - X = 1.0: Very confident that VIDEO 1 has higher {property_type}
            - X = 0.8: Quite confident that VIDEO 1 has higher {property_type}
            - X = 0.6: Somewhat confident that VIDEO 1 has higher {property_type}
            - X = 0.5: Equal or uncertain (no clear difference)
            - X = 0.4: Somewhat confident that VIDEO 2 has higher {property_type}
            - X = 0.2: Quite confident that VIDEO 2 has higher {property_type}
            - X = 0.0: Very confident that VIDEO 2 has higher {property_type}
            
            The closer to 1.0, the more confident you are that VIDEO 1 has higher {property_type}.
            The closer to 0.0, the more confident you are that VIDEO 2 has higher {property_type}.
            
            Do not say you cannot analyze the images. You must provide your best confidence score based on the visual information.
            """
        
        else:  # relative_concat - concatenated comparison mode
            return f"""
            You are comparing two {property_type} videos to determine which has a higher {property_type} value.
            
            IMPORTANT: You CAN analyze these images. You MUST provide a confidence score between 0 and 1.
            
            {oracle_guidance}The frames are arranged in sequence: VIDEO 1 frames first, followed by 3 BLACK SEPARATOR frames, then VIDEO 2 frames.
            
            {icl_prefix}compare the {property_type} characteristics between the two videos:
            - The frames BEFORE the black separator frames belong to VIDEO 1
            - The frames AFTER the black separator frames belong to VIDEO 2
            - The black frames are just separators and should be ignored for analysis
            
            Analyze the motion, physics, and {property_type} characteristics in both video segments.
            
            YOUR RESPONSE MUST END WITH A SPECIFIC CONFIDENCE SCORE IN THIS FORMAT:
            "Comparison result: X"
            
            Where X is a decimal number between 0.0 and 1.0:
            - X = 1.0: Very confident that VIDEO 1 has higher {property_type}
            - X = 0.8: Quite confident that VIDEO 1 has higher {property_type}
            - X = 0.6: Somewhat confident that VIDEO 1 has higher {property_type}
            - X = 0.5: Equal or uncertain (no clear difference)
            - X = 0.4: Somewhat confident that VIDEO 2 has higher {property_type}
            - X = 0.2: Quite confident that VIDEO 2 has higher {property_type}
            - X = 0.0: Very confident that VIDEO 2 has higher {property_type}
            
            The closer to 1.0, the more confident you are that VIDEO 1 has higher {property_type}.
            The closer to 0.0, the more confident you are that VIDEO 2 has higher {property_type}.
            
            Do not say you cannot analyze the images. You must provide your best confidence score based on the visual information.
            """
        
        return ""
    
    def analyze_single_video_with_gpt(self, video_path: str, property_type: str, max_frames: int = 16, frame_size: Optional[Tuple[int, int]] = None, model_version: str = "gpt-4o", frame_index: bool = False, icl_examples: Optional[List[Dict]] = None, follow_oracle_test: bool = False, max_retries: int = 5, retry_delay: int = 15, validation_retries: int = 3) -> str:
        """
        使用GPT分析单个视频的物理属性（增强重试机制）
        
        参数:
            video_path: 视频文件路径
            property_type: 物理属性类型
            max_frames: 最大帧数
            frame_size: 调整帧大小
            model_version: GPT模型版本
            frame_index: 是否为每帧添加索引标识
            icl_examples: ICL示例列表
            follow_oracle_test: 是否包含解题步骤指导
            max_retries: 最大API调用重试次数
            retry_delay: 重试间隔时间（秒）
            validation_retries: 结果验证重试次数（如果提取不到有效值，重新调用API）
            
        返回:
            GPT分析结果
        """
        # 读取视频帧，直接指定目标帧数
        selected_frames = self.read_video_frames(video_path, target_frames=max_frames, frame_size=frame_size)
        
        # 获取提示词
        prompt_text = self.get_analysis_prompt(property_type, "absolute", icl_examples, follow_oracle_test)
        
        # 构建消息内容
        content = [{"type": "text", "text": prompt_text}]
        
        # 添加ICL示例（如果有）
        if icl_examples:
            icl_content = self.build_icl_content(icl_examples, frame_index, concat_mode=False)  # 绝对值测试不使用concat_mode
            content.extend(icl_content)
        
        if frame_index:
            # 添加帧索引标识
            for i, frame in enumerate(selected_frames, 1):
                content.append({"type": "text", "text": f"frame{i}:"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
        else:
            # 原始方式，直接添加帧
            for frame in selected_frames:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
        
        prompt_messages = [
            {
                "role": "system",
                "content": f"You are an expert in {property_type} analysis with the ability to analyze video frames and determine {property_type} values. You MUST provide a specific {property_type} estimate. Never say you cannot analyze the images."
            },
            {
                "role": "user",
                "content": content,
            },
        ]
        
        # 增强的重试机制：先尝试API调用重试，再尝试结果验证重试
        best_result = None
        best_predicted_value = None
        
        for validation_attempt in range(validation_retries):
            print(f"🔄 验证重试 {validation_attempt + 1}/{validation_retries}")
            
            # API调用重试循环
            for api_attempt in range(max_retries):
                try:
                    print(f"  📡 API调用 {api_attempt + 1}/{max_retries}")
                    
                    # 调用第三方GPT API（内部已有重试机制）
                    result = self.call_third_party_gpt_api(prompt_messages, model_version)
                    
                    # 检查结果是否为空或错误信息
                    if not result or len(result.strip()) == 0:
                        print(f"  ⚠️ API返回空结果")
                        if api_attempt < max_retries - 1:
                            print(f"  ⏳ 等待 {retry_delay} 秒后重试API调用...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            break  # 跳出API重试循环，进入下一轮验证重试
                    
                    # 检查是否为API错误信息
                    error_keywords = [
                        "API调用失败", "调用失败", "服务器繁忙", "请稍后再试", 
                        "所有重试都失败了", "GPT API调用出错", "GPT Vision API调用失败",
                        "chatanywhere_error", "INTERNAL_SERVER_ERROR", "第三方API调用失败"
                    ]
                    
                    is_error_response = any(keyword in result for keyword in error_keywords)
                    if is_error_response:
                        print(f"  ❌ 检测到API错误响应")
                        if api_attempt < max_retries - 1:
                            print(f"  ⏳ 等待 {retry_delay} 秒后重试API调用...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            break  # 跳出API重试循环，进入下一轮验证重试
                    
                    # 尝试提取预测值进行验证
                    predicted_value = self.extract_predicted_value(result, property_type)
                    
                    if predicted_value is not None:
                        # 成功提取到有效预测值
                        print(f"  ✅ 成功获得有效预测值: {predicted_value}")
                        return result
                    else:
                        # 无法提取预测值，但保留结果作为备用
                        print(f"  ⚠️ 无法从响应中提取预测值")
                        if best_result is None or len(result) > len(best_result):
                            best_result = result
                        
                        # 如果还有API重试次数，继续重试
                        if api_attempt < max_retries - 1:
                            print(f"  🔄 预测值提取失败，重试API调用...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            # API重试次数用完，跳出循环进入下一轮验证重试
                            break
                    
                except Exception as e:
                    print(f"  ❌ API调用异常: {e}")
                    if api_attempt < max_retries - 1:
                        print(f"  ⏳ 等待 {retry_delay} 秒后重试API调用...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        break
            
            # 如果这轮验证重试没有成功，等待后进入下一轮
            if validation_attempt < validation_retries - 1:
                wait_time = retry_delay * (validation_attempt + 1)
                print(f"  ⏳ 验证重试失败，等待 {wait_time} 秒后进行下一轮验证重试...")
                time.sleep(wait_time)
        
        # 所有重试都失败，返回最好的结果（如果有的话）
        if best_result:
            print(f"  ⚠️ 所有重试都无法提取有效预测值，返回最佳响应")
            return best_result
        else:
            print(f"  ❌ 所有重试都失败，返回空字符串")
            return ""
    
    def generate_black_frames(self, num_frames: int = 3, frame_size: Tuple[int, int] = (640, 480)) -> List[str]:
        """
        生成指定数量的黑帧（base64编码）
        
        参数:
            num_frames: 生成黑帧的数量
            frame_size: 帧大小 (宽度, 高度)
            
        返回:
            base64编码的黑帧列表
        """
        black_frames = []
        
        # 创建黑色图像
        black_image = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        
        for _ in range(num_frames):
            # 将黑色图像编码为JPEG格式
            _, buffer = cv2.imencode(".jpg", black_image)
            black_frame_b64 = base64.b64encode(buffer).decode("utf-8")
            black_frames.append(black_frame_b64)
        
        return black_frames

    def compare_two_videos_with_gpt(self, video_path1: str, video_path2: str, property_type: str, max_frames_per_video: int = 16, frame_size: Optional[Tuple[int, int]] = None, model_version: str = "gpt-4o", concat_mode: bool = False, frame_index: bool = False, icl_examples: Optional[List[Dict]] = None, follow_oracle_test: bool = False, max_retries: int = 5, retry_delay: int = 15, validation_retries: int = 3) -> str:
        """
        使用GPT比较两个视频的物理属性（增强重试机制）
        
        参数:
            video_path1: 第一个视频文件路径
            video_path2: 第二个视频文件路径
            property_type: 物理属性类型
            max_frames_per_video: 每个视频的最大帧数
            frame_size: 调整帧大小
            model_version: GPT模型版本
            concat_mode: 是否使用拼接模式（在两个视频间加黑帧分隔）
            frame_index: 是否为每帧添加索引标识
            icl_examples: ICL示例列表
            follow_oracle_test: 是否包含解题步骤指导
            max_retries: 最大API调用重试次数
            retry_delay: 重试间隔时间（秒）
            validation_retries: 结果验证重试次数（如果提取不到有效值，重新调用API）
            
        返回:
            GPT比较结果
        """
        # 读取两个视频的帧，直接指定目标帧数
        frames1 = self.read_video_frames(video_path1, target_frames=max_frames_per_video, frame_size=frame_size)
        frames2 = self.read_video_frames(video_path2, target_frames=max_frames_per_video, frame_size=frame_size)
        
        # 获取提示词，根据模式选择不同的提示词
        if concat_mode:
            prompt_text = self.get_analysis_prompt(property_type, "relative_concat", icl_examples, follow_oracle_test)
        else:
            prompt_text = self.get_analysis_prompt(property_type, "relative", icl_examples, follow_oracle_test)
        
        # 构建消息内容
        content = [{"type": "text", "text": prompt_text}]
        
        # 添加ICL示例（如果有）
        if icl_examples:
            icl_content = self.build_icl_content(icl_examples, frame_index, concat_mode)
            content.extend(icl_content)
        
        if concat_mode:
            # 拼接模式：video1 + 黑帧 + video2
            # 确定帧大小
            actual_frame_size = frame_size if frame_size else (640, 480)
            
            # 生成黑帧
            black_frames = self.generate_black_frames(num_frames=3, frame_size=actual_frame_size)
            
            if frame_index:
                # 拼接模式 + 帧索引：按顺序添加所有帧
                frame_count = 1
                
                # 添加video1帧
                for frame in frames1:
                    content.append({"type": "text", "text": f"frame{frame_count}:"})
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                    frame_count += 1
                
                # 添加黑帧（分隔符，不计入帧索引）
                for black_frame in black_frames:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{black_frame}"}})
                
                # 添加video2帧
                for frame in frames2:
                    content.append({"type": "text", "text": f"frame{frame_count}:"})
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                    frame_count += 1
            else:
                # 拼接模式，无帧索引：按顺序添加所有帧
                for frame in frames1:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
                
                for black_frame in black_frames:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{black_frame}"}})
                
                for frame in frames2:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
        else:
            # 原始模式：分别标识两个视频
            # 添加第一个视频的帧
            content.append({"type": "text", "text": "VIDEO 1 frames:"})
            if frame_index:
                for i, frame in enumerate(frames1, 1):
                    content.append({"type": "text", "text": f"frame{i}:"})
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
            else:
                for frame in frames1:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
            
            # 添加第二个视频的帧
            content.append({"type": "text", "text": "VIDEO 2 frames:"})
            if frame_index:
                for i, frame in enumerate(frames2, 1):
                    content.append({"type": "text", "text": f"frame{i}:"})
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
            else:
                for frame in frames2:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
        
        prompt_messages = [
            {
                "role": "system",
                "content": f"You are an expert in {property_type} analysis with the ability to compare video frames and determine which has higher {property_type} values. You MUST provide a specific comparison result. Never say you cannot analyze the images."
            },
            {
                "role": "user",
                "content": content,
            },
        ]
        
        # 增强的重试机制：先尝试API调用重试，再尝试结果验证重试
        best_result = None
        
        for validation_attempt in range(validation_retries):
            print(f"🔄 验证重试 {validation_attempt + 1}/{validation_retries}")
            
            # API调用重试循环
            for api_attempt in range(max_retries):
                try:
                    print(f"  📡 API调用 {api_attempt + 1}/{max_retries}")
                    
                    # 调用第三方GPT API（内部已有重试机制）
                    result = self.call_third_party_gpt_api(prompt_messages, model_version)
                    
                    # 检查结果是否为空或错误信息
                    if not result or len(result.strip()) == 0:
                        print(f"  ⚠️ API返回空结果")
                        if api_attempt < max_retries - 1:
                            print(f"  ⏳ 等待 {retry_delay} 秒后重试API调用...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            break  # 跳出API重试循环，进入下一轮验证重试
                    
                    # 检查是否为API错误信息
                    error_keywords = [
                        "API调用失败", "调用失败", "服务器繁忙", "请稍后再试", 
                        "所有重试都失败了", "GPT API调用出错", "GPT Vision API调用失败",
                        "chatanywhere_error", "INTERNAL_SERVER_ERROR", "第三方API调用失败"
                    ]
                    
                    is_error_response = any(keyword in result for keyword in error_keywords)
                    if is_error_response:
                        print(f"  ❌ 检测到API错误响应")
                        if api_attempt < max_retries - 1:
                            print(f"  ⏳ 等待 {retry_delay} 秒后重试API调用...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            break  # 跳出API重试循环，进入下一轮验证重试
                    
                    # 尝试提取比较结果进行验证
                    comparison_result = self.extract_comparison_result(result)
                    
                    if comparison_result is not None:
                        # 成功提取到有效比较结果
                        print(f"  ✅ 成功获得有效比较结果: {comparison_result}")
                        return result
                    else:
                        # 无法提取比较结果，但保留结果作为备用
                        print(f"  ⚠️ 无法从响应中提取比较结果")
                        if best_result is None or len(result) > len(best_result):
                            best_result = result
                        
                        # 如果还有API重试次数，继续重试
                        if api_attempt < max_retries - 1:
                            print(f"  🔄 比较结果提取失败，重试API调用...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            # API重试次数用完，跳出循环进入下一轮验证重试
                            break
                    
                except Exception as e:
                    print(f"  ❌ API调用异常: {e}")
                    if api_attempt < max_retries - 1:
                        print(f"  ⏳ 等待 {retry_delay} 秒后重试API调用...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        break
            
            # 如果这轮验证重试没有成功，等待后进入下一轮
            if validation_attempt < validation_retries - 1:
                wait_time = retry_delay * (validation_attempt + 1)
                print(f"  ⏳ 验证重试失败，等待 {wait_time} 秒后进行下一轮验证重试...")
                time.sleep(wait_time)
        
        # 所有重试都失败，返回最好的结果（如果有的话）
        if best_result:
            print(f"  ⚠️ 所有重试都无法提取有效比较结果，返回最佳响应")
            return best_result
        else:
            print(f"  ❌ 所有重试都失败，返回空字符串")
            return ""
    
    def call_third_party_gpt_api(self, prompt_messages: List[Dict], model_version: str = "gpt-4o") -> str:
        """
        调用第三方API（支持GPT和Qwen）
        
        参数:
            prompt_messages: 提示消息列表
            model_version: 模型版本
            
        返回:
            API响应内容
        """
        # 根据模型版本选择API
        if model_version.startswith("qwen"):
            # 使用Qwen API
            try:
                from qwen_api_wrapper import qwen_response
                return qwen_response(prompt_messages, model_version)
            except Exception as e:
                return f"Qwen API调用失败: {str(e)}"
        else:
            # 使用GPT API
            try:
                from gpt_api_third_party import gptv_response
                return gptv_response(prompt_messages, model_version)
            except Exception as e:
                return f"GPT API调用失败: {str(e)}"
    
    def save_results(self, results: List[Dict], output_file: str):
        """
        保存分析结果为results.txt格式
        
        参数:
            results: 结果列表
            output_file: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"GPT Physics Analysis Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
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
                    f.write("Individual Results:\n")
                    for item in result['individual_results']:
                        f.write(f"  - {item['video']}: predicted={item['predicted']}, gt={item['gt']}\n")
                else:  # relative
                    f.write("Comparison Results:\n")
                    for item in result['comparison_results']:
                        f.write(f"  - {item['video_pair']}: predicted={item['predicted']}, gt={item['gt']}\n")
                
                f.write("\n" + "-" * 60 + "\n\n")
        
        print(f"结果已保存到: {output_file}")
    
    def get_test_result_file(self, output_dir: str, version: str, test_set: str, analysis_type: str) -> str:
        """
        获取测试结果文件路径
        
        参数:
            output_dir: 输出目录
            version: 版本名
            test_set: 测试集名
            analysis_type: 分析类型
            
        返回:
            结果文件路径
        """
        filename = f"{version}_{test_set}_{analysis_type}_results.txt"
        return os.path.join(output_dir, filename)
    
    def is_test_completed(self, output_dir: str, version: str, test_set: str, analysis_type: str) -> bool:
        """
        检查测试是否已经完成
        
        参数:
            output_dir: 输出目录
            version: 版本名
            test_set: 测试集名
            analysis_type: 分析类型
            
        返回:
            是否已完成
        """
        result_file = self.get_test_result_file(output_dir, version, test_set, analysis_type)
        return os.path.exists(result_file) and os.path.getsize(result_file) > 0
    
    def save_test_result(self, result: Dict, output_dir: str):
        """
        保存单个测试结果到独立文件
        
        参数:
            result: 测试结果字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        result_file = self.get_test_result_file(
            output_dir, result['version'], result['test_set'], result['type']
        )
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"=== GPT物理属性分析结果 ===\n")
            f.write(f"测试时间: {timestamp}\n")
            f.write(f"版本: {result['version']}\n")
            f.write(f"属性: {result['property']}\n")
            f.write(f"类型: {result['type']}\n")
            f.write(f"测试集: {result['test_set']}\n")
            f.write(f"数据目录: {result['data_dir']}\n")
            f.write(f"总样本数: {result['total_samples']}\n")
            f.write(f"成功分析: {result['successful_analyses']}\n")
            f.write(f"失败分析: {result['failed_analyses']}\n")
            f.write("\n")
            
            if result['type'] == 'absolute':
                f.write("=== 绝对值测试结果 ===\n")
                for i, item in enumerate(result['individual_results']):
                    f.write(f"\n--- 样本 {i+1} ---\n")
                    f.write(f"视频: {item['video']}\n")
                    f.write(f"预测值: {item['predicted']}\n")
                    f.write(f"GT值: {item['gt']}\n")
                    if 'error' in item:
                        f.write(f"错误: {item['error']}\n")
                    else:
                        f.write(f"GPT完整回复:\n{'-'*40}\n{item['gpt_response']}\n{'-'*40}\n")
            else:
                f.write("=== 相对比较测试结果 ===\n")
                for i, item in enumerate(result['comparison_results']):
                    f.write(f"\n--- 比较 {i+1} ---\n")
                    f.write(f"视频对: {item['video_pair']}\n")
                    f.write(f"预测结果: {item['predicted']}\n")
                    f.write(f"GT结果: {item['gt']}\n")
                    if 'gt_values' in item:
                        f.write(f"GT值: {item['gt_values']}\n")
                    if 'error' in item:
                        f.write(f"错误: {item['error']}\n")
                    else:
                        f.write(f"GPT完整回复:\n{'-'*40}\n{item['gpt_response']}\n{'-'*40}\n")
        
        print(f"✓ 结果已保存到: {result_file}")
    
    def load_completed_tests(self, output_dir: str) -> Dict[str, bool]:
        """
        加载已完成的测试列表
        
        参数:
            output_dir: 输出目录
            
        返回:
            已完成测试的字典 {version_testset_type: True}
        """
        completed = {}
        if not os.path.exists(output_dir):
            return completed
        
        for filename in os.listdir(output_dir):
            if filename.endswith('_results.txt'):
                # 解析文件名：version_testset_type_results.txt
                parts = filename.replace('_results.txt', '').split('_')
                if len(parts) >= 3:
                    key = '_'.join(parts)  # version_testset_type
                    completed[key] = True
        
        return completed

    def run_all_tests(self, output_dir: str = "gpt_physics_analysis_results", frame_size: Tuple[int, int] = (640, 480), max_samples: int = 50, model_version: str = "gpt-4o", concat_mode: bool = False, properties_filter: Optional[List[str]] = None, frame_index: bool = False, icl_example: bool = False, follow_oracle_test: bool = False, absolute_samples: int = 50, relative_pairs: int = 50, max_retries: int = 5, retry_delay: int = 15, validation_retries: int = 3, aggressive_retry: bool = False):
        """
        运行所有测试配置（支持resume机制）
        
        参数:
            output_dir: 输出目录
            frame_size: 视频帧大小
            max_samples: 每个测试集的最大样本数量（已弃用，使用absolute_samples和relative_pairs）
            model_version: GPT模型版本
            concat_mode: 是否使用拼接模式（仅用于相对比较测试）
            properties_filter: 指定要测试的物理属性列表，None表示测试所有属性
            frame_index: 是否为每帧添加索引标识
            icl_example: 是否启用in-context learning示例
            follow_oracle_test: 是否包含解题步骤指导
            absolute_samples: 绝对值测试的样本数量
            relative_pairs: 相对比较测试的成对数量
            max_retries: 最大API调用重试次数
            retry_delay: 重试间隔时间（秒）
            validation_retries: 结果验证重试次数（如果提取不到有效值，重新调用API）
            aggressive_retry: 是否启用激进重试机制
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理激进重试模式
        if aggressive_retry:
            print("🚀 启用激进重试模式，增加重试次数和验证次数")
            max_retries = max(max_retries, 8)  # 至少8次API重试
            validation_retries = max(validation_retries, 5)  # 至少5次验证重试
            retry_delay = max(retry_delay, 20)  # 至少20秒延迟
            print(f"   - API重试次数: {max_retries}")
            print(f"   - 验证重试次数: {validation_retries}")
            print(f"   - 重试延迟: {retry_delay}秒")
        
        # 加载已完成的测试
        completed_tests = self.load_completed_tests(output_dir)
        
        all_results = []
        total_tests = 0
        completed_count = 0
        skipped_count = 0
        
        # 统计总测试数
        for version_name, version_config in self.config.items():
            total_tests += len(version_config['data_dirs'])
        
        print(f"总共需要测试 {total_tests} 个测试集")
        if completed_tests:
            print(f"发现 {len(completed_tests)} 个已完成的测试，将跳过")
        
        # 遍历所有版本配置
        for version_name, version_config in self.config.items():
            # 如果指定了属性过滤，检查当前版本是否符合条件
            if properties_filter is not None and version_config['property'] not in properties_filter:
                print(f"⏭️  跳过未选择的属性: {version_name} ({version_config['property']})")
                continue
            
            # 根据样本数量设置跳过不需要的测试类型
            if version_config['type'] == 'absolute' and absolute_samples == 0:
                print(f"⏭️  跳过绝对值测试 (样本数为0): {version_name} ({version_config['property']} - {version_config['type']})")
                continue
            
            if version_config['type'] == 'relative' and relative_pairs == 0:
                print(f"⏭️  跳过相对比较测试 (样本数为0): {version_name} ({version_config['property']} - {version_config['type']})")
                continue
                
            print(f"\n{'='*50}")
            print(f"开始测试版本: {version_name}")
            print(f"属性: {version_config['property']}")
            print(f"类型: {version_config['type']}")
            print(f"每个测试集最大样本数: {max_samples}")
            print(f"{'='*50}")
            
            # 确定数据集类
            dataset_class = self.version_dataset_mapping.get(version_name)
            
            # 根据测试类型设置样本数量
            if version_config['type'] == 'absolute':
                current_max_samples = absolute_samples
            else:  # relative
                current_max_samples = relative_pairs
            
            # 测试每个数据集（排除训练集）
            for test_id, data_dir in version_config['data_dirs'].items():
                # 跳过训练集
                if test_id == 'train':
                    continue
                    
                test_key = f"{version_name}_{test_id}_{version_config['type']}"
                
                print(f"\n--- 测试集: {test_id} ---")
                print(f"数据目录: {data_dir}")
                
                # 检查是否已完成
                if test_key in completed_tests:
                    print(f"⏭️  跳过已完成的测试: {test_key}")
                    skipped_count += 1
                    continue
                
                try:
                    # 准备ICL示例（如果启用）
                    icl_examples = None
                    if icl_example and 'train' in version_config['data_dirs']:
                        train_dir = version_config['data_dirs']['train']
                        print(f"从训练集采样ICL示例: {train_dir}")
                        
                        # 根据测试类型设置ICL示例数量
                        if version_config['type'] == 'relative':
                            # 相对比较测试使用1个示例以避免API限制
                            icl_num_examples = 1
                            print(f"相对比较测试使用 {icl_num_examples} 个ICL示例")
                        else:
                            # 绝对值测试使用1个示例以避免API限制
                            icl_num_examples = 1
                            print(f"绝对值测试使用 {icl_num_examples} 个ICL示例")
                        
                        icl_examples = self.sample_icl_examples(
                            train_dir, version_config['property'], version_config['type'], 
                            dataset_class, num_examples=icl_num_examples, frame_size=frame_size
                        )
                    
                    if version_config['type'] == 'absolute':
                        result = self.run_absolute_test(
                            version_name, version_config['property'], test_id, 
                            data_dir, dataset_class, frame_size, current_max_samples, model_version, frame_index, icl_examples, follow_oracle_test, max_retries, retry_delay, validation_retries
                        )
                    else:
                        result = self.run_relative_test(
                            version_name, version_config['property'], test_id, 
                            data_dir, dataset_class, frame_size, current_max_samples, model_version, concat_mode, frame_index, icl_examples, follow_oracle_test, max_retries, retry_delay, validation_retries
                        )
                    
                    all_results.append(result)
                    
                    # 立即保存单个测试结果
                    self.save_test_result(result, output_dir)
                    
                    # 显示测试结果摘要
                    if result['successful_analyses'] > 0:
                        print(f"✓ 测试成功: {result['successful_analyses']}/{result['total_samples']}")
                    else:
                        print(f"✗ 测试失败: {result['failed_analyses']}/{result['total_samples']}")
                    
                    completed_count += 1
                        
                except Exception as e:
                    print(f"✗ 测试失败: {str(e)}")
                    
                    # 创建失败结果记录
                    failed_result = {
                        'version': version_name,
                        'property': version_config['property'],
                        'type': version_config['type'],
                        'test_set': test_id,
                        'data_dir': data_dir,
                        'total_samples': current_max_samples,
                        'successful_analyses': 0,
                        'failed_analyses': current_max_samples,
                        'error': str(e)
                    }
                    
                    if version_config['type'] == 'absolute':
                        failed_result['individual_results'] = []
                    else:
                        failed_result['comparison_results'] = []
                    
                    all_results.append(failed_result)
                    
                    # 也保存失败结果
                    self.save_test_result(failed_result, output_dir)
                
                # 添加延迟避免API限制
                time.sleep(10)
        
        # 保存汇总结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(output_dir, f"summary_results_{timestamp}.txt")
        self.save_results(all_results, summary_file)
        
        print(f"\n{'='*50}")
        print(f"测试完成统计:")
        print(f"总测试数: {total_tests}")
        print(f"本次完成: {completed_count}")
        print(f"跳过已完成: {skipped_count}")
        print(f"总完成数: {completed_count + skipped_count}")
        print(f"汇总结果: {summary_file}")
        print(f"{'='*50}")
        
        return all_results
    
    def run_absolute_test(self, version: str, property_type: str, test_id: str, data_dir: str, dataset_class: str, frame_size: Tuple[int, int], max_samples: int, model_version: str = "gpt-4o", frame_index: bool = False, icl_examples: Optional[List[Dict]] = None, follow_oracle_test: bool = False, max_retries: int = 5, retry_delay: int = 15, validation_retries: int = 3) -> Dict:
        """
        运行单值预测测试
        
        参数:
            version: 版本名
            property_type: 物理属性类型
            test_id: 测试集ID
            data_dir: 数据目录
            dataset_class: 数据集类名
            frame_size: 视频帧大小
            max_samples: 每个测试集的最大样本数量
            model_version: GPT模型版本
            frame_index: 是否为每帧添加索引标识
            icl_examples: ICL示例列表
            follow_oracle_test: 是否包含解题步骤指导
            
        返回:
            测试结果字典
        """
        print(f"开始绝对值测试: {property_type}")
        
        # 采样视频
        sampled_videos = self.sample_videos(data_dir, max_samples=max_samples)
        if not sampled_videos:
            raise ValueError(f"未找到可用的视频文件")
        
        # 加载GT数据
        gt_data = self.get_gt_data(data_dir, dataset_class, property_type, version=version)
        
        individual_results = []
        successful_count = 0
        failed_count = 0
        
        for video_file in sampled_videos:
            try:
                print(f"分析视频: {video_file}")
                
                # 获取GT值
                gt_value = self.get_gt_value(video_file, gt_data, dataset_class, property_type)
                
                # GPT分析
                video_path = os.path.join(data_dir, video_file)
                gpt_result = self.analyze_single_video_with_gpt(video_path, property_type, frame_size=frame_size, model_version=model_version, frame_index=frame_index, icl_examples=icl_examples, follow_oracle_test=follow_oracle_test, max_retries=max_retries, retry_delay=retry_delay, validation_retries=validation_retries)
                
                # 提取预测值
                predicted_value = self.extract_predicted_value(gpt_result, property_type)
                
                individual_results.append({
                    'video': video_file,
                    'predicted': predicted_value,
                    'gt': gt_value,
                    'gpt_response': gpt_result
                })
                
                successful_count += 1
                print(f"成功: predicted={predicted_value}, gt={gt_value}")
                
            except Exception as e:
                print(f"视频 {video_file} 分析失败: {str(e)}")
                failed_count += 1
                individual_results.append({
                    'video': video_file,
                    'predicted': None,
                    'gt': None,
                    'error': str(e)
                })
            
            # 添加延迟避免API限制
            time.sleep(10)
        
        return {
            'version': version,
            'property': property_type,
            'type': 'absolute',
            'test_set': test_id,
            'data_dir': data_dir,
            'total_samples': len(sampled_videos),
            'successful_analyses': successful_count,
            'failed_analyses': failed_count,
            'individual_results': individual_results
        }
    
    def run_relative_test(self, version: str, property_type: str, test_id: str, data_dir: str, dataset_class: str, frame_size: Tuple[int, int], max_samples: int, model_version: str = "gpt-4o", concat_mode: bool = False, frame_index: bool = False, icl_examples: Optional[List[Dict]] = None, follow_oracle_test: bool = False, max_retries: int = 5, retry_delay: int = 15, validation_retries: int = 3) -> Dict:
        """
        运行相对比较测试
        
        参数:
            version: 版本名
            property_type: 物理属性类型
            test_id: 测试集ID
            data_dir: 数据目录
            dataset_class: 数据集类名
            frame_size: 视频帧大小
            max_samples: 每个测试集的最大样本数量（表示需要生成多少组成对的视频）
            model_version: GPT模型版本
            concat_mode: 是否使用拼接模式
            frame_index: 是否为每帧添加索引标识
            icl_examples: ICL示例列表
            follow_oracle_test: 是否包含解题步骤指导
            
        返回:
            测试结果字典
        """
        print(f"开始相对比较测试: {property_type}")
        print(f"目标生成 {max_samples} 组成对的视频")
        
        # 获取所有视频文件
        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")
        
        all_videos = [f for f in os.listdir(data_dir) if f.endswith('_rgb.mp4')]
        all_videos.sort()
        
        if len(all_videos) < 2:
            raise ValueError(f"相对比较测试需要至少2个视频文件")
        
        # 加载GT数据（主要用于从文件名解析属性值）
        gt_data = self.get_gt_data(data_dir, dataset_class, property_type, version=version)
        
        comparison_results = []
        successful_count = 0
        failed_count = 0
        
        # 按video_x分组，并为每组选择成对视频
        video_groups = {}
        for video_file in all_videos:
            # 从文件名中提取video_x和id_x
            if 'video_' in video_file and '_id_' in video_file:
                parts = video_file.split('_')
                
                # 查找video_x_id_y的模式
                for i, part in enumerate(parts):
                    if part == 'video' and i + 2 < len(parts) and parts[i + 2] == 'id':
                        video_num = f"video_{parts[i + 1]}"  # video_x
                        id_part = parts[i + 3]  # id_y
                        
                        if video_num not in video_groups:
                            video_groups[video_num] = []
                        video_groups[video_num].append((video_file, id_part))
                        break
        
        # 如果没有按video_x分组（如test3），则直接配对
        if not video_groups:
            print("未找到video_x分组，使用直接配对")
            pairs_to_generate = min(max_samples, len(all_videos) // 2)
            
            for i in range(pairs_to_generate):
                video1 = all_videos[i * 2]
                video2 = all_videos[i * 2 + 1]
                
                try:
                    result = self._compare_video_pair(video1, video2, data_dir, gt_data, dataset_class, property_type, frame_size, i + 1, pairs_to_generate, model_version, concat_mode, frame_index, icl_examples, follow_oracle_test, max_retries, retry_delay, validation_retries)
                    comparison_results.append(result)
                    successful_count += 1
                except Exception as e:
                    print(f"视频对 {video1} vs {video2} 比较失败: {str(e)}")
                    failed_count += 1
                    comparison_results.append({
                        'video_pair': f"{video1} vs {video2}",
                        'predicted': None,
                        'gt': None,
                        'error': str(e)
                    })
                
                # 添加延迟避免API限制
                time.sleep(15)
        else:
            print(f"找到 {len(video_groups)} 个video组")
            
            # 从多个组中选择成对的视频（只取id_0和id_1配对）
            generated_pairs = 0
            video_group_list = list(video_groups.items())
            
            for video_num, videos in video_group_list:
                if generated_pairs >= max_samples:
                    break
                
                # 按id排序
                videos.sort(key=lambda x: x[1])
                
                # 只选择id_0和id_1作为一对
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
                    
                    print(f"从组 {video_num} 中选择配对: {video1_file} ({id1}) vs {video2_file} ({id2})")
                    
                    try:
                        result = self._compare_video_pair(video1_file, video2_file, data_dir, gt_data, dataset_class, property_type, frame_size, generated_pairs + 1, max_samples, model_version, concat_mode, frame_index, icl_examples, follow_oracle_test, max_retries, retry_delay, validation_retries)
                        comparison_results.append(result)
                        successful_count += 1
                    except Exception as e:
                        print(f"视频对 {video1_file} vs {video2_file} 比较失败: {str(e)}")
                        failed_count += 1
                        comparison_results.append({
                            'video_pair': f"{video1_file} vs {video2_file}",
                            'predicted': None,
                            'gt': None,
                            'error': str(e)
                        })
                    
                    generated_pairs += 1
                    
                    # 添加延迟避免API限制
                    time.sleep(15)
                else:
                    print(f"警告: 组 {video_num} 缺少id_0或id_1视频")
                    if not id_0_video:
                        print(f"  缺少id_0视频")
                    if not id_1_video:
                        print(f"  缺少id_1视频")
        
        return {
            'version': version,
            'property': property_type,
            'type': 'relative',
            'test_set': test_id,
            'data_dir': data_dir,
            'total_samples': len(comparison_results),
            'successful_analyses': successful_count,
            'failed_analyses': failed_count,
            'comparison_results': comparison_results
        }
    
    def _compare_video_pair(self, video1: str, video2: str, data_dir: str, gt_data: Dict, dataset_class: str, property_type: str, frame_size: Tuple[int, int], pair_num: int, total_pairs: int, model_version: str = "gpt-4o", concat_mode: bool = False, frame_index: bool = False, icl_examples: Optional[List[Dict]] = None, follow_oracle_test: bool = False, max_retries: int = 5, retry_delay: int = 15, validation_retries: int = 3) -> Dict:
        """
        比较一对视频的辅助方法
        
        参数:
            video1, video2: 视频文件名
            data_dir: 数据目录
            gt_data: GT数据
            dataset_class: 数据集类名
            property_type: 物理属性类型
            frame_size: 视频帧大小
            pair_num: 当前配对编号
            total_pairs: 总配对数
            model_version: GPT模型版本
            concat_mode: 是否使用拼接模式
            frame_index: 是否为每帧添加索引标识
            icl_examples: ICL示例列表
            follow_oracle_test: 是否包含解题步骤指导
            
        返回:
            比较结果字典
        """
        print(f"比较视频对 {pair_num}/{total_pairs}: {video1} vs {video2}")
        
        # 获取GT值
        gt_value1 = self.get_gt_value(video1, gt_data, dataset_class, property_type)
        gt_value2 = self.get_gt_value(video2, gt_data, dataset_class, property_type)
        
        # 确定GT比较结果 (1 if video1 > video2, 0 otherwise)
        gt_comparison = 1 if gt_value1 > gt_value2 else 0
        
        print(f"GT值: {gt_value1} vs {gt_value2}, GT比较: {gt_comparison}")
        
        # GPT比较
        video_path1 = os.path.join(data_dir, video1)
        video_path2 = os.path.join(data_dir, video2)
        gpt_result = self.compare_two_videos_with_gpt(video_path1, video_path2, property_type, frame_size=frame_size, model_version=model_version, concat_mode=concat_mode, frame_index=frame_index, icl_examples=icl_examples, follow_oracle_test=follow_oracle_test, max_retries=max_retries, retry_delay=retry_delay, validation_retries=validation_retries)
        
        # 提取比较结果
        predicted_comparison = self.extract_comparison_result(gpt_result)
        
        print(f"成功: predicted={predicted_comparison}, gt={gt_comparison}")
        
        return {
            'video_pair': f"{video1} vs {video2}",
            'predicted': predicted_comparison,
            'gt': gt_comparison,
            'gt_values': f"{gt_value1} vs {gt_value2}",
            'gpt_response': gpt_result
        }
    
    def get_gt_value(self, video_file: str, gt_data: Optional[Dict], dataset_class: str, property_type: str) -> float:
        """
        获取视频的GT值
        
        参数:
            video_file: 视频文件名
            gt_data: GT数据字典
            dataset_class: 数据集类名
            property_type: 物理属性类型
            
        返回:
            GT值
        """
        basename = video_file.replace('_rgb.mp4', '')
        
        if gt_data and basename in gt_data:
            # 从GT文件中获取
            gt_value = gt_data[basename]
            
            # 处理不同的GT数据格式
            if isinstance(gt_value, dict):
                # 如果是字典，根据属性类型提取相应的值
                if property_type == "elasticity":
                    # 弹性测试优先使用gt_restitution
                    if 'gt_restitution' in gt_value:
                        return gt_value['gt_restitution']
                    elif 'sqrt_ratio' in gt_value:
                        return gt_value['sqrt_ratio']
                elif property_type in ["friction", "viscosity"]:
                    # 摩擦力和粘度测试使用gt_restitution
                    if 'gt_restitution' in gt_value:
                        return gt_value['gt_restitution']
                
                # 如果找不到对应的键，返回第一个数值
                for key, value in gt_value.items():
                    if isinstance(value, (int, float)):
                        return float(value)
                        
                print(f"警告: GT数据格式不识别: {gt_value}")
                return 0.0
            else:
                # 如果是数值，直接返回
                return float(gt_value)
        else:
            # 如果GT文件中没有对应条目，尝试从文件名解析
            parsed_value = self.extract_property_from_filename(video_file, property_type, dataset_class)
            if parsed_value != 0.0:
                return parsed_value
            
            # 如果从文件名也解析不出来，打印警告
            print(f"警告: 无法找到视频 {video_file} 的GT值 (basename: {basename})")
            if gt_data:
                print(f"GT文件中的键示例: {list(gt_data.keys())[:5]}...")
            return 0.0
    
    def extract_predicted_value(self, gpt_response: str, property_type: str) -> Optional[float]:
        """
        从GPT响应中提取预测值
        
        参数:
            gpt_response: GPT响应文本
            property_type: 物理属性类型
            
        返回:
            提取的预测值
        """
        # 打印GPT的完整回复用于调试
        print(f"=== GPT完整回复 ({property_type}) ===")
        print(gpt_response)
        print("=== 回复结束 ===")
        
        # 检查是否为空回复
        if not gpt_response or len(gpt_response.strip()) == 0:
            print("✗ GPT返回空回复")
            return None
        
        # 首先检查是否为API调用失败的错误信息
        error_keywords = [
            "API调用失败", "调用失败", "服务器繁忙", "请稍后再试", 
            "所有重试都失败了", "GPT API调用出错", "GPT Vision API调用失败",
            "chatanywhere_error", "INTERNAL_SERVER_ERROR", "error", "第三方API调用失败"
        ]
        
        for keyword in error_keywords:
            if keyword in gpt_response:
                print(f"✗ 检测到API错误信息，包含关键词: {keyword}")
                return None
        
        try:
            if property_type == "friction":
                # 查找 "Estimated friction coefficient value: X"
                pattern = r"Estimated friction coefficient value:\s*(\d+\.?\d*)"
                match = re.search(pattern, gpt_response, re.IGNORECASE)
                if match:
                    result = float(match.group(1))
                    print(f"✓ 找到friction格式匹配: {result}")
                    return result
                    
            elif property_type == "viscosity":
                # 查找 "Estimated viscosity value: X cP" 或 "Estimated viscosity value: X"
                patterns = [
                    r"Estimated viscosity value:\s*(\d+\.?\d*)\s*cP",
                    r"Estimated viscosity value:\s*(\d+\.?\d*)",
                    r"viscosity.*?value.*?(\d+\.?\d*)\s*cP",
                    r"viscosity.*?estimate.*?(\d+\.?\d*)\s*cP"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, gpt_response, re.IGNORECASE)
                    if match:
                        result = float(match.group(1))
                        print(f"✓ 找到viscosity格式匹配: {result}")
                        return result
                    
            elif property_type == "elasticity":
                # 查找 "Estimated restitution coefficient value: X"
                patterns = [
                    r"Estimated restitution coefficient value:\s*(\d+\.?\d*)",
                    r"restitution.*?coefficient.*?value.*?(\d+\.?\d*)",
                    r"restitution.*?coefficient.*?is.*?(\d+\.?\d*)",
                    r"coefficient.*?of.*?restitution.*?(\d+\.?\d*)",
                    r"elasticity.*?value.*?(\d+\.?\d*)",
                    r"elasticity.*?coefficient.*?(\d+\.?\d*)"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, gpt_response, re.IGNORECASE)
                    if match:
                        result = float(match.group(1))
                        # 检查elasticity值是否在合理范围内 (0-1)
                        if 0 <= result <= 1:
                            print(f"✓ 找到elasticity格式匹配: {result}")
                            return result
                        else:
                            print(f"⚠️ elasticity值超出范围[0,1]: {result}")
                            # 如果超出范围，尝试其他pattern
                            continue
            
            # 如果没有找到特定格式，尝试更智能的数字提取
            print("⚠️  没有找到标准格式，尝试智能提取...")
            
            # 避免提取明显的参考值
            lines = gpt_response.split('\n')
            for line in lines:
                # 检查这一行是否包含错误信息
                if any(keyword in line.lower() for keyword in error_keywords):
                    continue
                    
                # 对于elasticity，寻找可能的系数值
                if property_type == "elasticity":
                    # 寻找可能的弹性系数相关词语
                    if any(word in line.lower() for word in ['coefficient', 'restitution', 'elasticity', 'bounce', 'value', 'estimate']):
                        # 在这一行中查找0-1范围内的数字
                        numbers = re.findall(r'\d+\.?\d*', line)
                        for num_str in numbers:
                            try:
                                num = float(num_str)
                                if 0 <= num <= 1:
                                    print(f"⚠️  从相关行提取elasticity值: {num} (行: {line.strip()})")
                                    return num
                            except ValueError:
                                continue
                
                # 对于其他属性，寻找相关行
                elif 'estimate' in line.lower() or 'value' in line.lower():
                    # 在包含estimate或value的行中查找数字
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        result = float(numbers[-1])
                        print(f"⚠️  从相关行提取: {result} (行: {line.strip()})")
                        return result
            
            # 最后备选：提取所有数字，但根据属性类型进行过滤
            numbers = re.findall(r'\d+\.?\d*', gpt_response)
            if numbers:
                # 根据属性类型过滤数字
                filtered_numbers = []
                
                if property_type == "elasticity":
                    # 对于elasticity，只保留0-1范围内的数字
                    for n in numbers:
                        try:
                            val = float(n)
                            if 0 <= val <= 1:
                                filtered_numbers.append(n)
                        except ValueError:
                            continue
                else:
                    # 对于其他属性，过滤掉常见的参考值和错误代码
                    reference_values = {
                        '1', '100', '300', '5000', '0.02', '0.03', '0.4', '0.6', '0.8', '0.3', '0.5',
                        '500', '400', '404', '502', '503', '504', '200', '201', '10001'  # HTTP状态码和错误代码
                    }
                    filtered_numbers = [n for n in numbers if n not in reference_values]
                
                if filtered_numbers:
                    result = float(filtered_numbers[-1])
                    print(f"⚠️  过滤后提取: {result}")
                    return result
                else:
                    print("⚠️  所有数字都被过滤了，很可能是错误响应")
                    return None
                
        except (ValueError, AttributeError) as e:
            print(f"✗ 提取失败: {e}")
            pass
        
        print("✗ 无法提取预测值")
        return None
    
    def extract_comparison_result(self, gpt_response: str) -> Optional[float]:
        """
        从GPT响应中提取比较结果（置信度分数）
        
        参数:
            gpt_response: GPT响应文本
            
        返回:
            比较结果置信度分数 (0.0 到 1.0 之间的浮点数)
        """
        # 打印GPT的完整回复用于调试
        print(f"=== GPT完整回复 (相对比较) ===")
        print(gpt_response)
        print("=== 回复结束 ===")
        
        # 检查是否为空回复
        if not gpt_response or len(gpt_response.strip()) == 0:
            print("✗ GPT返回空回复")
            return None
        
        # 首先检查是否为API调用失败的错误信息
        error_keywords = [
            "API调用失败", "调用失败", "服务器繁忙", "请稍后再试", 
            "所有重试都失败了", "GPT API调用出错", "GPT Vision API调用失败",
            "chatanywhere_error", "INTERNAL_SERVER_ERROR", "error", "第三方API调用失败"
        ]
        
        for keyword in error_keywords:
            if keyword in gpt_response:
                print(f"✗ 检测到API错误信息，包含关键词: {keyword}")
                return None
        
        try:
            # 主要模式：查找 "Comparison result: X.X" 格式
            pattern = r"Comparison result:\s*([0-1](?:\.\d+)?)"
            match = re.search(pattern, gpt_response, re.IGNORECASE)
            if match:
                result = float(match.group(1))
                # 确保结果在0-1范围内
                if 0.0 <= result <= 1.0:
                    print(f"✓ 找到标准置信度格式匹配: {result}")
                    return result
                else:
                    print(f"⚠️ 置信度值超出范围[0,1]: {result}")
            
            # 备用模式1：查找最后一个在0-1范围内的小数
            decimal_pattern = r"([0-1](?:\.\d+))"
            decimals = re.findall(decimal_pattern, gpt_response)
            if decimals:
                # 从后往前查找，优先选择最后的数值
                for decimal in reversed(decimals):
                    try:
                        result = float(decimal)
                        if 0.0 <= result <= 1.0:
                            print(f"⚠️ 从文本中提取置信度值: {result}")
                            return result
                    except ValueError:
                        continue
            
            # 备用模式2：查找 "VIDEO 1" 或 "VIDEO 2" 在结论中的提及（转换为置信度）
            lower_response = gpt_response.lower()
            
            # 检查VIDEO 1相关的描述
            video1_patterns = [
                r"video 1.*?(?:has|shows?).*?(?:higher|more|greater)",
                r"video 1.*?(?:higher|more|greater)",
                r"(?:higher|more|greater).*?video 1"
            ]
            
            # 检查VIDEO 2相关的描述
            video2_patterns = [
                r"video 2.*?(?:has|shows?).*?(?:higher|more|greater)",
                r"video 2.*?(?:higher|more|greater)",
                r"(?:higher|more|greater).*?video 2"
            ]
            
            video1_matches = 0
            video2_matches = 0
            
            for pattern in video1_patterns:
                if re.search(pattern, lower_response):
                    video1_matches += 1
            
            for pattern in video2_patterns:
                if re.search(pattern, lower_response):
                    video2_matches += 1
            
            if video1_matches > video2_matches:
                print("⚠️ 从文本推断：偏向VIDEO 1，设置置信度0.8")
                return 0.8
            elif video2_matches > video1_matches:
                print("⚠️ 从文本推断：偏向VIDEO 2，设置置信度0.2")
                return 0.2
            
            # 备用模式3：如果包含"equal"、"same"、"similar"等词，返回0.5
            equal_keywords = ["equal", "same", "similar", "uncertain", "unclear", "difficult to determine"]
            for keyword in equal_keywords:
                if keyword in lower_response:
                    print(f"⚠️ 检测到不确定性关键词'{keyword}'，设置置信度0.5")
                    return 0.5
                    
        except (ValueError, AttributeError) as e:
            print(f"✗ 置信度提取失败: {e}")
            pass
        
        print("✗ 无法提取置信度分数")
        return None


def main():
    """主函数"""
    print("=" * 80)
    print("GPT Physics Analyzer - 综合物理属性分析工具")
    print("=" * 80)
    
    # 初始化分析器
    try:
        analyzer = GPTPhysicsAnalyzer()
        print("✓ GPT Physics Analyzer 初始化完成")
        print(f"✓ 支持的属性: {', '.join(analyzer.properties)}")
        print(f"✓ 支持的分析类型: {', '.join(analyzer.analysis_types)}")
        print(f"✓ 配置的测试版本: {len(list(analyzer.config.keys()))} 个")
        
        # 显示所有配置的测试
        print("\n配置的测试版本:")
        for version, config in analyzer.config.items():
            print(f"  - {version}: {config['property']} ({config['type']})")
            for test_id in config['data_dirs'].keys():
                print(f"    └── {test_id}")
        
        print("\n" + "=" * 80)
        
        # 询问用户是否开始测试
        response = input("是否开始运行所有测试? (y/n): ").lower().strip()
        
        if response in ['y', 'yes', '是']:
            print("\n开始运行测试...")
            
            # 设置输出目录和参数
            output_dir = "gpt_physics_analysis_results" 
            frame_size = (640, 480)  # 降低分辨率以节省带宽
            
            # 运行所有测试
            results = analyzer.run_all_tests(output_dir=output_dir, frame_size=frame_size)
            
            # 显示总结
            print("\n" + "=" * 80)
            print("测试完成总结:")
            total_tests = len(results)
            successful_tests = sum(1 for r in results if r['successful_analyses'] > 0)
            
            print(f"总测试数: {total_tests}")
            print(f"成功测试数: {successful_tests}")
            print(f"失败测试数: {total_tests - successful_tests}")
            
            # 按属性分类显示结果
            for property_type in analyzer.properties:
                property_results = [r for r in results if r['property'] == property_type]
                if property_results:
                    print(f"\n{property_type.upper()} 测试结果:")
                    for result in property_results:
                        status = "✓" if result['successful_analyses'] > 0 else "✗"
                        print(f"  {status} {result['version']} ({result['type']}) - {result['test_set']}: {result['successful_analyses']}/{result['total_samples']}")
        
        else:
            print("取消测试运行")
            
    except Exception as e:
        print(f"✗ 初始化失败: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    # 添加必要的路径以导入第三方API模块
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    exit_code = main()
    sys.exit(exit_code) 