#!/usr/bin/env python3
"""
计算SVD模型在GPT测试子集上的指标
Author: AI Assistant
Date: 2025-01-XX
"""

import os
import json
import re
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SVDSubsetMetricsCalculator:
    """
    计算SVD模型在GPT测试子集上的指标
    """
    
    def __init__(self):
        """初始化配置"""
        self.gpt_results_dir = "test_mllms_from_api/gpt_physics_analysis_results"
        self.svd_results_dir = "scatter_plot_results_5.25"
        
        # SVD模型路径映射（基于实际存在的目录）
        self.svd_paths = {
            'v120': {
                'test1': '5.25_result_v120test1_375itr',
                'test2': '5.25_result_v120test2_375itr',
                'test3': '5.25_result_v120test3_375itr'
            },
            'v121': {
                'test1': '5.25_result_v121test1_2500itr',
                'test2': '5.25_result_v121test2_2500itr',
                'test3': '5.25_result_v121test3_2500itr'
            },
            'v122': {
                'test1': '5.25_result_v122test1_5125itr',
                'test2': '5.25_result_v122test2_5125itr',
                'test3': '5.25_result_v122test3_5125itr_16frames'
            },
            'v123': {
                'test1': '5.25_result_v123test1_1000itr',
                'test2': '5.25_result_v123test2_1000itr',
                'test3': '5.25_result_v123test3_1000itr'
            },
            'v124': {
                'test1': '5.25_result_v124_test1_3000itr',
                'test2': '5.25_result_v124_test2_3000itr',
                'test3': '5.25_result_v124_test3_3000itr'
            },
            'v125': {
                'test1': '5.25_result_v125_test1_1375itr',
                'test2': '5.25_result_v125_test2_1375itr',
                'test3': '5.25_result_v125_test3_1375itr'
            }
        }
        
        # SVD模型属性映射（基于对应关系）
        self.svd_properties = {
            'v120': 'friction',    # 对应原v120
            'v121': 'friction',    # 对应原v121
            'v122': 'viscosity',   # 对应原v119
            'v123': 'viscosity',   # 对应原v110
            'v124': 'elasticity',  # 对应原v124
            'v125': 'elasticity'   # 对应原v125
        }
        
        # SVD模型测试类型映射
        self.svd_test_types = {
            'v120': 'absolute',    # friction 绝对值测试
            'v121': 'relative',    # friction 相对比较测试
            'v122': 'absolute',    # viscosity 绝对值测试 (对应v119)
            'v123': 'relative',    # viscosity 相对比较测试 (对应v110)
            'v124': 'absolute',    # elasticity 绝对值测试
            'v125': 'relative'     # elasticity 相对比较测试
        }
        
        # SVD模型到GPT测试结果的映射
        self.svd_to_gpt_mapping = {
            'v120': 'v120',  # v120 SVD 对应 v120 GPT
            'v121': 'v121',  # v121 SVD 对应 v121 GPT
            'v122': 'v119',  # v122 SVD 对应 v119 GPT
            'v123': 'v110',  # v123 SVD 对应 v110 GPT
            'v124': 'v124',  # v124 SVD 对应 v124 GPT
            'v125': 'v125'   # v125 SVD 对应 v125 GPT
        }
        
        # 物理属性处理函数
        self.processing_functions = {
            'viscosity': self._process_viscosity,
            'friction': self._process_friction,
            'elasticity': self._process_elasticity
        }
        
        self.results = {}
    
    def _process_viscosity(self, pred: float, is_absolute: bool) -> float:
        """
        处理粘度预测值
        
        Args:
            pred: 原始预测值
            is_absolute: 是否是绝对值测试
            
        Returns:
            处理后的预测值
        """
        if is_absolute:
            # 绝对值：sigmoid(pred) * 0.01
            return self._sigmoid(pred) * 0.01
        else:
            # 相对值：pred > 0 ? 1 : 0
            return 1 if pred > 0 else 0
    
    def _process_friction(self, pred: float, is_absolute: bool) -> float:
        """
        处理摩擦力预测值
        
        Args:
            pred: 原始预测值
            is_absolute: 是否是绝对值测试
            
        Returns:
            处理后的预测值
        """
        if is_absolute:
            # 绝对值：sigmoid(pred) * 0.2
            return self._sigmoid(pred) * 0.2
        else:
            # 相对值：pred > 0 ? 1 : 0
            return 1 if pred > 0 else 0
    
    def _process_elasticity(self, pred: float, is_absolute: bool) -> float:
        """
        处理弹性预测值
        
        Args:
            pred: 原始预测值
            is_absolute: 是否是绝对值测试
            
        Returns:
            处理后的预测值
        """
        # 弹性直接使用原始值
        return pred
    
    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid函数
        
        Args:
            x: 输入值
            
        Returns:
            sigmoid(x)
        """
        return 1 / (1 + np.exp(-x))
    
    def _extract_video_id(self, video_name: str) -> str:
        """
        从完整的视频名称中提取video_X_id_Y格式的标识符
        
        Args:
            video_name: 完整的视频名称
            
        Returns:
            提取的video_X_id_Y格式标识符
        """
        # 使用正则表达式提取 video_X_id_Y 格式
        pattern = r'(video_\d+_id_\d+)'
        match = re.search(pattern, video_name)
        if match:
            return match.group(1)
        
        # 如果没有匹配到，返回原始名称
        print(f"警告：无法从 '{video_name}' 中提取video_X_id_Y格式")
        return video_name
    
    def _parse_gpt_absolute_results(self, filepath: str) -> Dict:
        """
        解析GPT绝对值测试结果
        
        Args:
            filepath: GPT结果文件路径
            
        Returns:
            解析后的结果字典
        """
        results = {'samples': []}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析视频名称和GT值
            # 修改正则表达式以匹配实际格式
            pattern = r'视频: ([^\n]+)\n.*?GT值: ([^\n]+)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for video_name, gt_str in matches:
                video_name = video_name.strip()
                try:
                    gt = float(gt_str.strip())
                    results['samples'].append({
                        'video': video_name,
                        'gt': gt
                    })
                except ValueError:
                    print(f"警告：无法解析GT值 '{gt_str}' 对于视频 {video_name}")
                    continue
                    
        except Exception as e:
            print(f"解析GPT绝对值结果失败 {filepath}: {e}")
        
        return results
    
    def _parse_gpt_relative_results(self, filepath: str) -> Dict:
        """
        解析GPT相对比较测试结果
        
        Args:
            filepath: GPT结果文件路径
            
        Returns:
            解析后的结果字典
        """
        results = {'comparisons': []}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析视频对和GT结果
            pattern = r'视频对: ([^\n]+) vs ([^\n]+)\n.*?GT结果: ([^\n]+)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for video1, video2, gt_str in matches:
                video1 = video1.strip()
                video2 = video2.strip()
                try:
                    gt = int(gt_str.strip())
                    results['comparisons'].append({
                        'video1': video1,
                        'video2': video2,
                        'gt': gt
                    })
                except ValueError:
                    print(f"警告：无法解析GT值 '{gt_str}' 对于比较 {video1} vs {video2}")
                    continue
                    
        except Exception as e:
            print(f"解析GPT相对比较结果失败 {filepath}: {e}")
        
        return results
    
    def _extract_svd_absolute_data(self, version: str, video_names: List[str], test_name: str) -> Dict:
        """
        从SVD结果中提取绝对值测试数据
        
        Args:
            version: 模型版本
            video_names: 需要提取的视频名称列表
            test_name: 测试名称 (test1, test2, test3)
            
        Returns:
            包含提取数据的字典
        """
        results = {
            'version': version,
            'samples': []
        }
        
        # 获取SVD结果路径
        if version not in self.svd_paths:
            print(f"警告：未找到版本 {version} 的路径映射")
            return results
        
        # 从测试名称中提取test编号
        test_number = test_name.replace('test', '')
        test_key = f'test{test_number}'
        
        if test_key not in self.svd_paths[version]:
            print(f"警告：未找到版本 {version} 的 {test_key} 路径映射")
            return results
        
        result_path = self.svd_paths[version][test_key]
        full_path = os.path.join(self.svd_results_dir, result_path)
        
        if not os.path.exists(full_path):
            print(f"警告：未找到SVD结果目录 {full_path}")
            return results
        
        try:
            # 创建视频名称到数据的映射
            svd_data = {}
            
            # 遍历所有JSON文件
            for filename in os.listdir(full_path):
                if filename.endswith('.json'):
                    json_path = os.path.join(full_path, filename)
                    
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 处理JSON数据
                        for key, value in data.items():
                            if 'pred' in value and 'gt' in value:
                                svd_data[key] = {
                                    'pred': value['pred'],
                                    'gt': value['gt']
                                }
                    except Exception as e:
                        print(f"读取SVD JSON文件失败 {json_path}: {e}")
                        continue
            
            # 提取对应的样本数据
            found_count = 0
            not_found = []
            
            for video_name in video_names:
                matched = False
                
                # 策略1: 直接匹配
                if video_name in svd_data:
                    results['samples'].append({
                        'video': video_name,
                        'pred': svd_data[video_name]['pred'],
                        'gt': svd_data[video_name]['gt']
                    })
                    matched = True
                    found_count += 1
                
                if not matched:
                    # 策略2: 去掉扩展名匹配
                    video_base = video_name.replace('.mp4', '')
                    for svd_key in svd_data.keys():
                        svd_base = svd_key.replace('.mp4', '')
                        if video_base == svd_base:
                            results['samples'].append({
                                'video': video_name,
                                'pred': svd_data[svd_key]['pred'],
                                'gt': svd_data[svd_key]['gt']
                            })
                            matched = True
                            found_count += 1
                            break
                
                if not matched:
                    # 策略3: 更灵活的匹配（处理浮点数格式差异）
                    video_base = video_name.replace('.mp4', '')
                    video_normalized = video_base.replace('-0.00', '_0.00').replace('_0.00', '_0.0')
                    for svd_key in svd_data.keys():
                        svd_base = svd_key.replace('.mp4', '')
                        svd_normalized = svd_base.replace('-0.00', '_0.00').replace('_0.00', '_0.0')
                        if video_normalized == svd_normalized or video_base in svd_base or svd_base in video_base:
                            results['samples'].append({
                                'video': video_name,
                                'pred': svd_data[svd_key]['pred'],
                                'gt': svd_data[svd_key]['gt']
                            })
                            matched = True
                            found_count += 1
                            break
                
                if not matched:
                    # 策略4: 基于ID的匹配
                    # 从视频名称中提取ID
                    id_match = re.search(r'id_(\d+)', video_name)
                    if id_match:
                        video_id = id_match.group(1)
                        for svd_key in svd_data.keys():
                            if f'id_{video_id}' in svd_key:
                                results['samples'].append({
                                    'video': video_name,
                                    'pred': svd_data[svd_key]['pred'],
                                    'gt': svd_data[svd_key]['gt']
                                })
                                matched = True
                                found_count += 1
                                break
                
                if not matched:
                    not_found.append(video_name)
            
            # 打印匹配统计信息
            print(f"提取SVD绝对值数据：{version}，总视频数：{len(video_names)}，匹配成功：{found_count}，未匹配：{len(not_found)}")
            if len(not_found) > 0 and len(not_found) <= 5:  # 只显示前5个未匹配的
                print(f"  未匹配的视频示例：{not_found[:5]}")
            
            # 调试信息：显示SVD数据中的视频数量
            print(f"  SVD数据库中共有 {len(svd_data)} 个视频")
            
        except Exception as e:
            print(f"提取SVD绝对值数据失败 {full_path}: {e}")
        
        return results
    
    def _extract_svd_relative_data(self, version: str, video_pairs: List[Tuple[str, str]], test_name: str) -> Dict:
        """
        从SVD结果中提取相对比较测试数据
        
        Args:
            version: 模型版本
            video_pairs: 需要提取的视频对列表
            test_name: 测试名称 (test1, test2, test3)
            
        Returns:
            包含提取数据的字典
        """
        results = {
            'version': version,
            'comparisons': []
        }
        
        # 获取SVD结果路径
        if version not in self.svd_paths:
            print(f"警告：未找到版本 {version} 的路径映射")
            return results
        
        # 从测试名称中提取test编号
        test_number = test_name.replace('test', '')
        test_key = f'test{test_number}'
        
        if test_key not in self.svd_paths[version]:
            print(f"警告：未找到版本 {version} 的 {test_key} 路径映射")
            return results
        
        result_path = self.svd_paths[version][test_key]
        full_path = os.path.join(self.svd_results_dir, result_path)
        
        if not os.path.exists(full_path):
            print(f"警告：未找到SVD结果目录 {full_path}")
            return results
        
        try:
            # 首先将GPT测试中的视频对转换为简短标识符
            target_pairs = []
            for video1, video2 in video_pairs:
                id1 = self._extract_video_id(video1)
                id2 = self._extract_video_id(video2)
                target_pairs.append((id1, id2))
            
            # 遍历所有json文件
            for filename in os.listdir(full_path):
                if filename.endswith('.json'):
                    json_path = os.path.join(full_path, filename)
                    
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 解析json数据
                        for key, value in data.items():
                            if '--' in key:
                                parts = key.split('--')
                                if len(parts) == 2:
                                    video1, video2 = parts
                                    # 检查是否是我们需要的视频对
                                    for target_id1, target_id2 in target_pairs:
                                        if (video1 == target_id1 and video2 == target_id2) or \
                                           (video1 == target_id2 and video2 == target_id1):
                                            results['comparisons'].append({
                                                'video1': video1,
                                                'video2': video2,
                                                'pred': value['pred'],
                                                'gt': value['gt']
                                            })
                                            break
                    except Exception as e:
                        print(f"读取SVD JSON文件失败 {json_path}: {e}")
                        continue
            
            print(f"提取SVD相对比较数据：{version}，比较数：{len(results['comparisons'])}")
            
        except Exception as e:
            print(f"提取SVD相对比较数据失败 {full_path}: {e}")
        
        return results
    
    def _calculate_pearson(self, preds: List[float], gts: List[float]) -> Tuple[float, float]:
        """
        计算Pearson相关系数
        
        Args:
            preds: 预测值列表
            gts: 真实值列表
            
        Returns:
            (correlation, p_value)
        """
        try:
            correlation, p_value = stats.pearsonr(preds, gts)
            return correlation, p_value
        except Exception as e:
            print(f"计算Pearson相关系数失败: {e}")
            return 0.0, 1.0
    
    def _calculate_roc_auc(self, preds: List[float], gts: List[int]) -> float:
        """
        计算ROC-AUC
        
        Args:
            preds: 预测值列表
            gts: 真实值列表
            
        Returns:
            ROC-AUC值
        """
        try:
            return roc_auc_score(gts, preds)
        except Exception as e:
            print(f"计算ROC-AUC失败: {e}")
            return 0.0
    
    def calculate_metrics(self):
        """
        计算所有SVD模型的指标
        """
        print("开始计算SVD模型在GPT测试子集上的指标...")
        
        # 为每个SVD版本计算指标
        for version, property_name in self.svd_properties.items():
            test_type = self.svd_test_types[version]
            gpt_version = self.svd_to_gpt_mapping[version]  # 获取对应的GPT版本
            print(f"\n=== 处理SVD模型 {version} ({property_name}, {test_type}) -> GPT {gpt_version} ===")
            
            # 寻找对应的GPT测试结果
            gpt_files = []
            for filename in os.listdir(self.gpt_results_dir):
                if filename.startswith(f'{gpt_version}_') and filename.endswith('.txt'):
                    # 检查文件是否符合测试类型
                    if test_type == 'absolute' and 'absolute' in filename:
                        gpt_files.append(filename)
                    elif test_type == 'relative' and 'relative' in filename:
                        gpt_files.append(filename)
            
            if not gpt_files:
                print(f"警告：未找到版本 {gpt_version} 的 {test_type} GPT测试结果")
                continue
            
            # 处理每个测试文件
            for gpt_file in gpt_files:
                gpt_path = os.path.join(self.gpt_results_dir, gpt_file)
                
                if test_type == 'absolute':
                    # 绝对值测试
                    gpt_data = self._parse_gpt_absolute_results(gpt_path)
                    print(f"解析绝对值测试结果：{gpt_file.replace('.txt', '')}，样本数：{len(gpt_data['samples'])}")
                    
                    # 提取视频名称
                    video_names = [sample['video'] for sample in gpt_data['samples']]
                    
                    # 从SVD结果中提取对应数据
                    # 提取测试名称，例如从 'v120_vjepa_test1_absolute_results.txt' 中提取 'test1'
                    test_name_full = gpt_file.replace('.txt', '').replace(f'{gpt_version}_vjepa_', '')
                    test_name = test_name_full.split('_')[0]  # 提取 'test1', 'test2', 'test3'
                    svd_data = self._extract_svd_absolute_data(version, video_names, test_name)
                    
                    if len(svd_data['samples']) > 0:
                        # 应用处理逻辑
                        processed_preds = []
                        processed_gts = []
                        
                        for sample in svd_data['samples']:
                            processed_pred = self.processing_functions[property_name](sample['pred'], True)
                            processed_preds.append(processed_pred)
                            processed_gts.append(sample['gt'])
                        
                        # 计算Pearson相关系数
                        correlation, p_value = self._calculate_pearson(processed_preds, processed_gts)
                        
                        print(f"  绝对值测试 {property_name}: 样本数={len(processed_preds)}, Pearson={correlation:.4f}")
                        
                        # 保存结果
                        test_name = gpt_file.replace('.txt', '').replace(f'{gpt_version}_', '')
                        result_key = f"{version}_{property_name}_{test_name}"
                        self.results[result_key] = {
                            'version': version,
                            'property': property_name,
                            'test_type': 'absolute',
                            'test_name': test_name,
                            'samples': len(processed_preds),
                            'pearson': correlation,
                            'p_value': p_value
                        }
                    
                elif test_type == 'relative':
                    # 相对比较测试
                    gpt_data = self._parse_gpt_relative_results(gpt_path)
                    print(f"解析相对比较测试结果：{gpt_file.replace('.txt', '')}，比较数：{len(gpt_data['comparisons'])}")
                    
                    # 提取视频对
                    video_pairs = [(comp['video1'], comp['video2']) for comp in gpt_data['comparisons']]
                    
                    # 从SVD结果中提取对应数据
                    # 提取测试名称，例如从 'v121_vjepa_test1_relative_results.txt' 中提取 'test1'
                    test_name_full = gpt_file.replace('.txt', '').replace(f'{gpt_version}_vjepa_', '')
                    test_name = test_name_full.split('_')[0]  # 提取 'test1', 'test2', 'test3'
                    svd_data = self._extract_svd_relative_data(version, video_pairs, test_name)
                    
                    if len(svd_data['comparisons']) > 0:
                        # 应用处理逻辑
                        processed_preds = []
                        processed_gts = []
                        
                        for comp in svd_data['comparisons']:
                            processed_pred = self.processing_functions[property_name](comp['pred'], False)
                            processed_preds.append(processed_pred)
                            processed_gts.append(comp['gt'])
                        
                        # 计算ROC-AUC
                        roc_auc = self._calculate_roc_auc(processed_preds, processed_gts)
                        
                        print(f"  相对比较测试 {property_name}: 比较数={len(processed_preds)}, ROC-AUC={roc_auc:.4f}")
                        
                        # 保存结果
                        test_name = gpt_file.replace('.txt', '').replace(f'{gpt_version}_', '')
                        result_key = f"{version}_{property_name}_{test_name}"
                        self.results[result_key] = {
                            'version': version,
                            'property': property_name,
                            'test_type': 'relative',
                            'test_name': test_name,
                            'comparisons': len(processed_preds),
                            'roc_auc': roc_auc
                        }
    
    def save_results(self):
        """
        保存结果到文件
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果到JSON
        json_filename = f"svd_subset_metrics_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 生成文本报告
        report_filename = f"svd_subset_metrics_report_{timestamp}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SVD模型在GPT测试子集上的指标计算结果\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # 绝对值测试结果
            f.write("=== 绝对值测试结果 (Pearson相关系数) ===\n")
            f.write(f"{'版本':<10} {'属性':<12} {'测试集':<20} {'样本数':<8} {'Pearson':<10} {'P值':<10}\n")
            f.write("-" * 80 + "\n")
            
            for key, result in self.results.items():
                if result['test_type'] == 'absolute':
                    f.write(f"{result['version']:<10} {result['property']:<12} {result['test_name']:<20} "
                           f"{result['samples']:<8} {result['pearson']:<10.4f} "
                           f"{result['p_value']:<10.4f}\n")
            
            f.write("\n")
            
            # 相对比较测试结果
            f.write("=== 相对比较测试结果 (ROC-AUC) ===\n")
            f.write(f"{'版本':<10} {'属性':<12} {'测试集':<20} {'比较数':<8} {'ROC-AUC':<10}\n")
            f.write("-" * 60 + "\n")
            
            for key, result in self.results.items():
                if result['test_type'] == 'relative':
                    f.write(f"{result['version']:<10} {result['property']:<12} {result['test_name']:<20} "
                           f"{result['comparisons']:<8} {result['roc_auc']:<10.4f}\n")
            
            f.write("\n")
            
            # 按属性分组的统计
            f.write("=== 按属性分组的平均结果 ===\n")
            
            properties = set(result['property'] for result in self.results.values())
            for prop in sorted(properties):
                prop_results = [r for r in self.results.values() if r['property'] == prop]
                absolute_results = [r for r in prop_results if r['test_type'] == 'absolute']
                relative_results = [r for r in prop_results if r['test_type'] == 'relative']
                
                avg_pearson = np.mean([r['pearson'] for r in absolute_results]) if absolute_results else 0
                avg_roc_auc = np.mean([r['roc_auc'] for r in relative_results]) if relative_results else 0
                
                f.write(f"{prop:<12} 平均Pearson: {avg_pearson:.4f}, 平均ROC-AUC: {avg_roc_auc:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("报告生成完成\n")
            f.write("=" * 80 + "\n")
        
        print(f"报告已保存到 {report_filename}")
        print(f"详细结果已保存到 {json_filename}")
        
        return report_filename, json_filename
    
    def generate_summary(self):
        """
        生成结果摘要
        """
        print("\n" + "=" * 50)
        print("计算完成！结果摘要：")
        print("=" * 50)
        
        # 统计结果
        total_absolute = sum(1 for r in self.results.values() if r['test_type'] == 'absolute')
        total_relative = sum(1 for r in self.results.values() if r['test_type'] == 'relative')
        
        print(f"总计完成：{total_absolute} 个绝对值测试，{total_relative} 个相对比较测试")
        
        # 按属性分组显示结果
        properties = set(result['property'] for result in self.results.values())
        for prop in sorted(properties):
            print(f"\n{prop.upper()} 测试结果：")
            prop_results = [r for r in self.results.values() if r['property'] == prop]
            
            for result in prop_results:
                if result['test_type'] == 'absolute':
                    print(f"  {result['version']} {result['test_name']}: Pearson={result['pearson']:.4f} (样本数={result['samples']})")
                else:
                    print(f"  {result['version']} {result['test_name']}: ROC-AUC={result['roc_auc']:.4f} (比较数={result['comparisons']})")


def main():
    """
    主函数
    """
    calculator = SVDSubsetMetricsCalculator()
    
    # 计算指标
    calculator.calculate_metrics()
    
    # 保存结果
    report_file, json_file = calculator.save_results()
    
    # 生成摘要
    calculator.generate_summary()
    
    return report_file, json_file


if __name__ == "__main__":
    main() 