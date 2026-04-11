#!/usr/bin/env python3
"""
计算vjepa模型在GPT测试子集上的指标
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


class VjepaSubsetMetricsCalculator:
    """
    计算vjepa模型在GPT测试子集上的指标
    """
    
    def __init__(self):
        """初始化配置"""
        self.gpt_results_dir = "test_mllms_from_api/gpt_physics_analysis_results"
        self.vjepa_results_dir = "scatter_plot_results_5.25"
        
        # vjepa模型路径映射
        self.vjepa_paths = {
            'v110': '5.25_result_v110_vjepa_test1_20000itr',
            'v119': '5.25_result_v119vjepa_test1_25000itr',
            'v120': '5.25_result_v120vjepa_test1_25000itr',
            'v121': '5.25_result_v121_vjepa_test1_2000itr-v2',
            'v124': '5.25_result_v124_vjepa_test1_3250itr',
            'v125': '5.25_result_v125_vjepa_test1_2250itr'
        }
        
        # 特殊路径映射（处理一些路径的特殊后缀）
        self.special_paths = {
            'v119_test3': '5.25_result_v119vjepa_test3_25000itr_16frames',
        }
        
        # 物理属性处理函数
        self.processing_functions = {
            'viscosity': self._process_viscosity,
            'friction': self._process_friction,
            'elasticity': self._process_elasticity
        }
        
        self.results = {}
    
    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid激活函数
        
        Args:
            x: 输入值
            
        Returns:
            Sigmoid变换后的值
        """
        try:
            return 1 / (1 + np.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    def _process_viscosity(self, pred: float, is_absolute: bool = True) -> float:
        """
        处理粘度预测值
        
        Args:
            pred: 预测值
            is_absolute: 是否为绝对值测试
            
        Returns:
            处理后的预测值
        """
        if is_absolute:
            return self._sigmoid(pred) * 0.01
        else:
            return 1.0 if pred > 0 else 0.0
    
    def _process_friction(self, pred: float, is_absolute: bool = True) -> float:
        """
        处理摩擦系数预测值
        
        Args:
            pred: 预测值
            is_absolute: 是否为绝对值测试
            
        Returns:
            处理后的预测值
        """
        if is_absolute:
            return self._sigmoid(pred) * 0.2
        else:
            return 1.0 if pred > 0 else 0.0
    
    def _process_elasticity(self, pred: float, is_absolute: bool = True) -> float:
        """
        处理弹性模量预测值
        
        Args:
            pred: 预测值
            is_absolute: 是否为绝对值测试
            
        Returns:
            处理后的预测值（直接返回）
        """
        return pred
    
    def _parse_gpt_absolute_results(self, file_path: str) -> Dict:
        """
        解析GPT绝对值测试结果
        
        Args:
            file_path: GPT结果文件路径
            
        Returns:
            包含样本信息的字典
        """
        results = {
            'version': '',
            'property': '',
            'test_set': '',
            'samples': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取基本信息
            version_match = re.search(r'版本: (\w+)', content)
            property_match = re.search(r'属性: (\w+)', content)
            test_set_match = re.search(r'测试集: (\w+)', content)
            
            if version_match:
                results['version'] = version_match.group(1)
            if property_match:
                results['property'] = property_match.group(1)
            if test_set_match:
                results['test_set'] = test_set_match.group(1)
            
            # 提取样本信息
            sample_pattern = r'--- 样本 \d+ ---\s*视频: ([^\s]+)\s*预测值: ([^\s]+)\s*GT值: ([^\s]+)'
            samples = re.findall(sample_pattern, content)
            
            for video_name, pred_str, gt_str in samples:
                try:
                    pred = float(pred_str)
                    gt = float(gt_str)
                    results['samples'].append({
                        'video': video_name,
                        'pred': pred,
                        'gt': gt
                    })
                except ValueError:
                    print(f"警告：无法解析样本数据 {video_name}")
                    continue
            
            print(f"解析绝对值测试结果：{results['version']} {results['property']} {results['test_set']}，样本数：{len(results['samples'])}")
            
        except Exception as e:
            print(f"解析GPT绝对值结果文件失败 {file_path}: {e}")
        
        return results
    
    def _parse_gpt_relative_results(self, file_path: str) -> Dict:
        """
        解析GPT相对比较测试结果
        
        Args:
            file_path: GPT结果文件路径
            
        Returns:
            包含比较信息的字典
        """
        results = {
            'version': '',
            'property': '',
            'test_set': '',
            'comparisons': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取基本信息
            version_match = re.search(r'版本: (\w+)', content)
            property_match = re.search(r'属性: (\w+)', content)
            test_set_match = re.search(r'测试集: (\w+)', content)
            
            if version_match:
                results['version'] = version_match.group(1)
            if property_match:
                results['property'] = property_match.group(1)
            if test_set_match:
                results['test_set'] = test_set_match.group(1)
            
            # 提取比较信息
            comparison_pattern = r'--- 比较 \d+ ---\s*视频对: ([^\s]+) vs ([^\s]+)\s*预测结果: ([^\s]+)\s*GT结果: ([^\s]+)'
            comparisons = re.findall(comparison_pattern, content)
            
            for video1, video2, pred_str, gt_str in comparisons:
                try:
                    pred = int(pred_str)
                    gt = int(gt_str)
                    results['comparisons'].append({
                        'video1': video1,
                        'video2': video2,
                        'pred': pred,
                        'gt': gt
                    })
                except ValueError:
                    print(f"警告：无法解析比较数据 {video1} vs {video2}")
                    continue
            
            print(f"解析相对比较测试结果：{results['version']} {results['property']} {results['test_set']}，比较数：{len(results['comparisons'])}")
            
        except Exception as e:
            print(f"解析GPT相对比较结果文件失败 {file_path}: {e}")
        
        return results
    
    def _get_vjepa_path(self, version: str, test_set: str) -> str:
        """
        获取vjepa结果路径，处理特殊情况
        
        Args:
            version: 模型版本
            test_set: 测试集名称
            
        Returns:
            vjepa结果路径
        """
        # 检查特殊路径映射
        special_key = f"{version}_{test_set}"
        if special_key in self.special_paths:
            return self.special_paths[special_key]
        
        # 使用常规路径映射
        if version not in self.vjepa_paths:
            return ""
        
        base_path = self.vjepa_paths[version]
        # 替换test1为对应的test_set
        result_path = base_path.replace('test1', test_set)
        
        return result_path
    
    def _extract_vjepa_absolute_data(self, version: str, test_set: str, video_names: List[str]) -> Dict:
        """
        从vjepa全集结果中提取绝对值测试数据
        
        Args:
            version: 模型版本
            test_set: 测试集名称
            video_names: 需要提取的视频名称列表
            
        Returns:
            包含提取数据的字典
        """
        results = {
            'version': version,
            'test_set': test_set,
            'samples': []
        }
        
        # 获取vjepa结果路径
        result_path = self._get_vjepa_path(version, test_set)
        if not result_path:
            print(f"警告：未找到版本 {version} 的路径映射")
            return results
        
        full_path = os.path.join(self.vjepa_results_dir, result_path)
        
        # 查找results.txt文件
        results_file = os.path.join(full_path, 'results.txt')
        if not os.path.exists(results_file):
            print(f"警告：未找到vjepa结果文件 {results_file}")
            return results
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析results.txt中的数据
            file_pattern = r'File: ([^,]+), Prediction: ([^,]+), GroundTruth: ([^\s]+)'
            matches = re.findall(file_pattern, content)
            
            # 创建查找字典
            vjepa_data = {}
            for filename, pred_str, gt_str in matches:
                try:
                    pred = float(pred_str)
                    gt = float(gt_str)
                    vjepa_data[filename] = {'pred': pred, 'gt': gt}
                except ValueError:
                    continue
            
            # 提取对应的样本数据
            for video_name in video_names:
                if video_name in vjepa_data:
                    results['samples'].append({
                        'video': video_name,
                        'pred': vjepa_data[video_name]['pred'],
                        'gt': vjepa_data[video_name]['gt']
                    })
                else:
                    print(f"警告：在vjepa结果中未找到视频 {video_name}")
            
            print(f"提取vjepa绝对值数据：{version} {test_set}，样本数：{len(results['samples'])}")
            
        except Exception as e:
            print(f"提取vjepa绝对值数据失败 {full_path}: {e}")
        
        return results
    
    def _extract_video_id(self, video_name: str) -> str:
        """
        从完整的视频名称中提取video_X_id_Y格式的标识符
        
        Args:
            video_name: 完整的视频名称
            
        Returns:
            提取的video_X_id_Y格式标识符
        """
        # 使用正则表达式提取 video_X_id_Y 格式（包含更多数字）
        # 匹配 video_数字_id_数字 的模式
        pattern = r'(video_\d+_id_\d+)'
        match = re.search(pattern, video_name)
        if match:
            return match.group(1)
        
        # 如果没有匹配到，尝试旧的模式作为后备
        pattern_old = r'(video_\d+_id_\d+)'
        match_old = re.search(pattern_old, video_name)
        if match_old:
            return match_old.group(1)
            
        # 如果都没有匹配到，返回原始名称
        print(f"警告：无法从 '{video_name}' 中提取video_X_id_Y格式")
        return video_name
    
    def _extract_vjepa_relative_data(self, version: str, test_set: str, video_pairs: List[Tuple[str, str]]) -> Dict:
        """
        从vjepa全集结果中提取相对比较测试数据
        
        Args:
            version: 模型版本
            test_set: 测试集名称
            video_pairs: 需要提取的视频对列表
            
        Returns:
            包含提取数据的字典
        """
        results = {
            'version': version,
            'test_set': test_set,
            'comparisons': []
        }
        
        # 获取vjepa结果路径
        result_path = self._get_vjepa_path(version, test_set)
        if not result_path:
            print(f"警告：未找到版本 {version} 的路径映射")
            return results
        
        full_path = os.path.join(self.vjepa_results_dir, result_path)
        
        if not os.path.exists(full_path):
            print(f"警告：未找到vjepa结果目录 {full_path}")
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
                        print(f"读取json文件失败 {json_path}: {e}")
                        continue
            
            print(f"提取vjepa相对比较数据：{version} {test_set}，比较数：{len(results['comparisons'])}")
            
        except Exception as e:
            print(f"提取vjepa相对比较数据失败 {full_path}: {e}")
        
        return results
    
    def _calculate_pearson_correlation(self, predictions: List[float], ground_truths: List[float]) -> Tuple[float, float]:
        """
        计算Pearson相关系数
        
        Args:
            predictions: 预测值列表
            ground_truths: 真实值列表
            
        Returns:
            (相关系数, p值)
        """
        if len(predictions) != len(ground_truths) or len(predictions) < 2:
            return 0.0, 1.0
        
        try:
            corr, p_val = stats.pearsonr(predictions, ground_truths)
            return corr, p_val
        except Exception as e:
            print(f"计算Pearson相关系数失败: {e}")
            return 0.0, 1.0
    
    def _calculate_roc_auc(self, predictions: List[float], ground_truths: List[int]) -> float:
        """
        计算ROC-AUC分数
        
        Args:
            predictions: 预测值列表
            ground_truths: 真实标签列表
            
        Returns:
            ROC-AUC分数
        """
        if len(predictions) != len(ground_truths) or len(predictions) < 2:
            return 0.0
        
        try:
            # 检查标签是否有变化
            if len(set(ground_truths)) < 2:
                print("警告：标签无变化，无法计算ROC-AUC")
                return 0.0
            
            auc = roc_auc_score(ground_truths, predictions)
            return auc
        except Exception as e:
            print(f"计算ROC-AUC失败: {e}")
            return 0.0
    
    def process_model_results(self, version: str):
        """
        处理单个模型的所有结果
        
        Args:
            version: 模型版本
        """
        print(f"\n=== 处理模型 {version} ===")
        
        model_results = {
            'version': version,
            'absolute': {},
            'relative': {}
        }
        
        # 处理绝对值测试
        for test_set in ['test1', 'test2', 'test3']:
            for property_name in ['viscosity', 'friction', 'elasticity']:
                # 查找GPT结果文件
                gpt_file = f"{version}_vjepa_{test_set}_absolute_results.txt"
                gpt_path = os.path.join(self.gpt_results_dir, gpt_file)
                
                if os.path.exists(gpt_path):
                    # 解析GPT结果
                    gpt_results = self._parse_gpt_absolute_results(gpt_path)
                    
                    if gpt_results['samples'] and gpt_results['property'] == property_name:
                        # 提取视频名称
                        video_names = [sample['video'] for sample in gpt_results['samples']]
                        
                        # 从vjepa全集中提取对应数据
                        vjepa_data = self._extract_vjepa_absolute_data(version, test_set, video_names)
                        
                        if vjepa_data['samples']:
                            # 处理预测值
                            processed_preds = []
                            gts = []
                            
                            for sample in vjepa_data['samples']:
                                processed_pred = self.processing_functions[property_name](sample['pred'], is_absolute=True)
                                processed_preds.append(processed_pred)
                                gts.append(sample['gt'])
                            
                            # 计算Pearson相关系数
                            corr, p_val = self._calculate_pearson_correlation(processed_preds, gts)
                            
                            # 存储结果
                            key = f"{property_name}_{test_set}"
                            model_results['absolute'][key] = {
                                'property': property_name,
                                'test_set': test_set,
                                'sample_count': len(vjepa_data['samples']),
                                'pearson_corr': corr,
                                'p_value': p_val
                            }
                            
                            print(f"  绝对值测试 {property_name} {test_set}: 样本数={len(vjepa_data['samples'])}, Pearson={corr:.4f}")
        
        # 处理相对比较测试
        for test_set in ['test1', 'test2', 'test3']:
            for property_name in ['viscosity', 'friction', 'elasticity']:
                # 查找GPT结果文件
                gpt_file = f"{version}_vjepa_{test_set}_relative_results.txt"
                gpt_path = os.path.join(self.gpt_results_dir, gpt_file)
                
                if os.path.exists(gpt_path):
                    # 解析GPT结果
                    gpt_results = self._parse_gpt_relative_results(gpt_path)
                    
                    if gpt_results['comparisons'] and gpt_results['property'] == property_name:
                        # 提取视频对
                        video_pairs = [(comp['video1'], comp['video2']) for comp in gpt_results['comparisons']]
                        
                        # 从vjepa全集中提取对应数据
                        vjepa_data = self._extract_vjepa_relative_data(version, test_set, video_pairs)
                        
                        if vjepa_data['comparisons']:
                            # 处理预测值
                            processed_preds = []
                            gts = []
                            
                            for comp in vjepa_data['comparisons']:
                                processed_pred = self.processing_functions[property_name](comp['pred'], is_absolute=False)
                                processed_preds.append(processed_pred)
                                gts.append(comp['gt'])
                            
                            # 计算ROC-AUC
                            auc = self._calculate_roc_auc(processed_preds, gts)
                            
                            # 存储结果
                            key = f"{property_name}_{test_set}"
                            model_results['relative'][key] = {
                                'property': property_name,
                                'test_set': test_set,
                                'comparison_count': len(vjepa_data['comparisons']),
                                'roc_auc': auc
                            }
                            
                            print(f"  相对比较测试 {property_name} {test_set}: 比较数={len(vjepa_data['comparisons'])}, ROC-AUC={auc:.4f}")
        
        self.results[version] = model_results
    
    def generate_report(self) -> str:
        """
        生成对比报告
        
        Returns:
            报告字符串
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append(f"VJEPA模型在GPT测试子集上的指标计算结果")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        
        # 绝对值测试结果汇总
        report_lines.append("\n=== 绝对值测试结果 (Pearson相关系数) ===")
        report_lines.append(f"{'版本':<10} {'属性':<12} {'测试集':<8} {'样本数':<8} {'Pearson':<10} {'P值':<10}")
        report_lines.append("-" * 70)
        
        for version in sorted(self.results.keys()):
            model_results = self.results[version]
            for key, result in sorted(model_results['absolute'].items()):
                report_lines.append(f"{version:<10} {result['property']:<12} {result['test_set']:<8} "
                                  f"{result['sample_count']:<8} {result['pearson_corr']:<10.4f} {result['p_value']:<10.4f}")
        
        # 相对比较测试结果汇总
        report_lines.append("\n=== 相对比较测试结果 (ROC-AUC) ===")
        report_lines.append(f"{'版本':<10} {'属性':<12} {'测试集':<8} {'比较数':<8} {'ROC-AUC':<10}")
        report_lines.append("-" * 60)
        
        for version in sorted(self.results.keys()):
            model_results = self.results[version]
            for key, result in sorted(model_results['relative'].items()):
                report_lines.append(f"{version:<10} {result['property']:<12} {result['test_set']:<8} "
                                  f"{result['comparison_count']:<8} {result['roc_auc']:<10.4f}")
        
        # 按属性分组的平均结果
        report_lines.append("\n=== 按属性分组的平均结果 ===")
        
        # 计算平均值
        property_stats = {}
        for version in self.results.keys():
            model_results = self.results[version]
            
            # 绝对值测试平均值
            for key, result in model_results['absolute'].items():
                prop = result['property']
                if prop not in property_stats:
                    property_stats[prop] = {'absolute_pearson': [], 'relative_auc': []}
                property_stats[prop]['absolute_pearson'].append(result['pearson_corr'])
            
            # 相对比较测试平均值
            for key, result in model_results['relative'].items():
                prop = result['property']
                if prop not in property_stats:
                    property_stats[prop] = {'absolute_pearson': [], 'relative_auc': []}
                property_stats[prop]['relative_auc'].append(result['roc_auc'])
        
        for prop, stats in sorted(property_stats.items()):
            avg_pearson = np.mean(stats['absolute_pearson']) if stats['absolute_pearson'] else 0.0
            avg_auc = np.mean(stats['relative_auc']) if stats['relative_auc'] else 0.0
            
            report_lines.append(f"{prop:<12} 平均Pearson: {avg_pearson:.4f}, 平均ROC-AUC: {avg_auc:.4f}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("报告生成完成")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_detailed_results(self, filename: str):
        """
        保存详细结果到JSON文件
        
        Args:
            filename: 输出文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"详细结果已保存到 {filename}")
        except Exception as e:
            print(f"保存详细结果失败: {e}")
    
    def run(self):
        """
        运行完整的计算流程
        """
        print("开始计算vjepa模型在GPT测试子集上的指标...")
        
        # 处理所有模型版本
        for version in self.vjepa_paths.keys():
            try:
                self.process_model_results(version)
            except Exception as e:
                print(f"处理模型 {version} 时发生错误: {e}")
                continue
        
        # 生成报告
        report = self.generate_report()
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"vjepa_subset_metrics_report_{timestamp}.txt"
        
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n报告已保存到 {report_filename}")
        except Exception as e:
            print(f"保存报告失败: {e}")
        
        # 保存详细结果
        detailed_filename = f"vjepa_subset_metrics_{timestamp}.json"
        self.save_detailed_results(detailed_filename)
        
        # 打印报告摘要
        print("\n" + "=" * 50)
        print("计算完成！结果摘要：")
        print("=" * 50)
        print(report)


if __name__ == "__main__":
    calculator = VjepaSubsetMetricsCalculator()
    calculator.run() 