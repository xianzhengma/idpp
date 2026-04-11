#!/usr/bin/env python3
"""
重测失败预测脚本 - Elasticity Relative版本
用于重新测试elasticity relative ablation实验中的无效预测，
并将重测结果合并回原始结果文件
"""

import os
import glob
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from gpt_physics_analyzer import GPTPhysicsAnalyzer


class RetryFailedPredictionsRelative:
    """重测失败预测的处理器 - Relative版本"""
    
    def __init__(self):
        """初始化重测处理器"""
        self.analyzer = GPTPhysicsAnalyzer()
        self.property_type = "elasticity"
        self.analysis_type = "relative"
        
        # 实验配置映射
        self.experiment_configs = {
            'gemini_physics_analysis_results_elasticity_relative_20samples': {
                'frame_index': False,
                'icl_examples': None,
                'follow_oracle_test': False,
                'concat_mode': False
            },
            'gemini_physics_analysis_results_elasticity_relative_20samples_frameindex': {
                'frame_index': True,
                'icl_examples': None,
                'follow_oracle_test': False,
                'concat_mode': False
            },
            'gemini_physics_analysis_results_elasticity_relative_20samples_icl': {
                'frame_index': False,
                'icl_examples': [],  # 会在运行时生成
                'follow_oracle_test': False,
                'concat_mode': False
            },
            'gemini_physics_analysis_results_elasticity_relative_20samples_oracle': {
                'frame_index': False,
                'icl_examples': None,
                'follow_oracle_test': True,
                'concat_mode': False
            },
            'gemini_physics_analysis_results_elasticity_relative_20samples_concat': {
                'frame_index': False,
                'icl_examples': None,
                'follow_oracle_test': False,
                'concat_mode': True
            },
            'gemini_physics_analysis_results_relative_100samples_oracle': {
                'frame_index': False,
                'icl_examples': None,
                'follow_oracle_test': True,
                'concat_mode': False
            }
        }
    
    def parse_result_file(self, result_file: str) -> List[Dict]:
        """
        解析relative结果文件，提取所有比较信息
        
        参数:
            result_file: 结果文件路径
            
        返回:
            比较信息列表
        """
        samples = []
        
        with open(result_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有比较结果
        pattern = r'--- 比较 (\d+) ---\n.*?视频对: (.*?)\n.*?预测结果: (.*?)\n.*?GT结果: (.*?)\n.*?GT值: (.*?)\n.*?GPT完整回复:\n-{40}\n(.*?)\n-{40}'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            comparison_id, video_pair, predicted_str, gt_str, gt_values, gpt_response = match
            
            # 解析预测结果
            predicted = None
            if predicted_str != 'None' and predicted_str.strip():
                try:
                    predicted = float(predicted_str)
                except ValueError:
                    predicted = None
            
            # 解析GT结果
            gt = None
            if gt_str != 'None' and gt_str.strip():
                try:
                    gt = int(gt_str)
                except ValueError:
                    gt = None
            
            # 解析视频对
            video_pair_parts = video_pair.split(' vs ')
            video1 = video_pair_parts[0].strip() if len(video_pair_parts) > 0 else ""
            video2 = video_pair_parts[1].strip() if len(video_pair_parts) > 1 else ""
            
            samples.append({
                'comparison_id': int(comparison_id),
                'video_pair': video_pair.strip(),
                'video1': video1,
                'video2': video2,
                'predicted': predicted,
                'gt': gt,
                'gt_values': gt_values.strip(),
                'gpt_response': gpt_response.strip(),
                'is_failed': predicted is None
            })
        
        return samples
    
    def get_data_directory_for_test(self, test_name: str) -> str:
        """
        根据测试名称获取对应的数据目录
        
        参数:
            test_name: 测试名称 (Test-1, Test-2, Test-3)
            
        返回:
            数据目录路径
        """
        test_mapping = {
            'Test-1': '/scratch/shared/beegfs/guanqi/genesis_data/5.24_elasticity_synthetic_test1',
            'Test-2': '/scratch/shared/beegfs/guanqi/genesis_data/5.24_elasticity_synthetic_test2',
            'Test-3': '/scratch/shared/beegfs/guanqi/genesis_data/2.16_elasticity_test3'
        }
        return test_mapping.get(test_name)
    
    def get_test_name_from_filename(self, filename: str) -> str:
        """
        从文件名提取测试名称
        
        参数:
            filename: 文件名
            
        返回:
            测试名称
        """
        if 'test1' in filename:
            return 'Test-1'
        elif 'test2' in filename:
            return 'Test-2'
        elif 'test3' in filename:
            return 'Test-3'
        else:
            return 'Unknown'
    
    def generate_icl_examples(self, data_dir: str, num_examples: int = 3) -> List[Dict]:
        """
        为ICL实验生成示例
        
        参数:
            data_dir: 数据目录
            num_examples: 示例数量
            
        返回:
            ICL示例列表
        """
        try:
            # 使用analyzer的方法生成ICL示例
            icl_examples = self.analyzer.sample_icl_examples(
                train_dir=data_dir,
                property_type=self.property_type,
                analysis_type=self.analysis_type,
                dataset_class="WebVidv3ElasticityTest",  # 对于relative elasticity使用这个类
                num_examples=num_examples,
                frame_size=(224, 224)
            )
            return icl_examples
        except Exception as e:
            print(f"生成ICL示例失败: {e}")
            return []
    
    def retry_failed_comparison(self, sample: Dict, data_dir: str, config: Dict) -> Optional[str]:
        """
        重测单个失败的比较
        
        参数:
            sample: 比较信息
            data_dir: 数据目录
            config: 实验配置
            
        返回:
            新的GPT响应或None
        """
        video1_path = os.path.join(data_dir, sample['video1'])
        video2_path = os.path.join(data_dir, sample['video2'])
        
        # 检查两个视频文件是否都存在
        if not os.path.exists(video1_path):
            print(f"  视频文件1不存在: {video1_path}")
            return None
        
        if not os.path.exists(video2_path):
            print(f"  视频文件2不存在: {video2_path}")
            return None
        
        try:
            # 准备ICL示例（如果需要）
            icl_examples = None
            if config['icl_examples'] is not None:
                if len(config['icl_examples']) == 0:
                    # 需要生成ICL示例
                    icl_examples = self.generate_icl_examples(data_dir, num_examples=3)
                else:
                    icl_examples = config['icl_examples']
            
            # 调用GPT进行比较分析
            gpt_result = self.analyzer.compare_two_videos_with_gpt(
                video_path1=video1_path,
                video_path2=video2_path,
                property_type=self.property_type,
                max_frames_per_video=16,
                frame_size=(224, 224),
                model_version="gpt-4o",
                concat_mode=config['concat_mode'],
                frame_index=config['frame_index'],
                icl_examples=icl_examples,
                follow_oracle_test=config['follow_oracle_test']
            )
            
            # 验证是否成功提取到比较结果
            predicted_comparison = self.analyzer.extract_comparison_result(gpt_result)
            
            if predicted_comparison is not None:
                print(f"  ✓ 重测成功: {predicted_comparison}")
                return gpt_result
            else:
                print(f"  ✗ 重测仍然失败，无法提取比较结果")
                return None
                
        except Exception as e:
            print(f"  ✗ 重测出错: {e}")
            return None
    
    def update_result_file(self, result_file: str, samples: List[Dict], updated_samples: Dict[int, str]) -> None:
        """
        更新结果文件，将重测成功的结果写回
        
        参数:
            result_file: 结果文件路径
            samples: 原始样本列表
            updated_samples: 更新的样本字典 {comparison_id: new_gpt_response}
        """
        if not updated_samples:
            print("  没有需要更新的样本")
            return
        
        # 创建备份
        backup_file = result_file + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(result_file, 'r', encoding='utf-8') as src, open(backup_file, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        print(f"  已创建备份: {backup_file}")
        
        # 读取原文件内容
        with open(result_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 更新每个成功重测的比较
        for comparison_id, new_gpt_response in updated_samples.items():
            # 提取新的比较结果
            new_predicted = self.analyzer.extract_comparison_result(new_gpt_response)
            
            if new_predicted is not None:
                # 查找并替换对应比较的内容
                pattern = rf'(--- 比较 {comparison_id} ---\n.*?视频对: .*?\n.*?)预测结果: None(\n.*?GT结果: .*?\n.*?GT值: .*?\n.*?GPT完整回复:\n-{{40}}\n).*?(\n-{{40}})'
                replacement = rf'\g<1>预测结果: {new_predicted}\g<2>{new_gpt_response}\g<3>'
                
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # 写回文件
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✓ 已更新 {len(updated_samples)} 个比较的结果")
    
    def process_experiment_directory(self, exp_dir: str) -> None:
        """
        处理单个实验目录
        
        参数:
            exp_dir: 实验目录路径
        """
        print(f"\n处理实验目录: {exp_dir}")
        
        # 获取实验配置
        exp_name = os.path.basename(exp_dir)
        if exp_name not in self.experiment_configs:
            print(f"  警告: 未知的实验配置: {exp_name}")
            return
        
        config = self.experiment_configs[exp_name]
        print(f"  配置: frame_index={config['frame_index']}, "
              f"icl={'是' if config['icl_examples'] is not None else '否'}, "
              f"oracle={'是' if config['follow_oracle_test'] else '否'}, "
              f"concat={'是' if config['concat_mode'] else '否'}")
        
        # 查找结果文件
        result_files = glob.glob(os.path.join(exp_dir, "*relative_results.txt"))
        
        if not result_files:
            print(f"  警告: 未找到结果文件")
            return
        
        total_retried = 0
        total_successful = 0
        
        # 处理每个结果文件
        for result_file in result_files:
            filename = os.path.basename(result_file)
            test_name = self.get_test_name_from_filename(filename)
            data_dir = self.get_data_directory_for_test(test_name)
            
            if not data_dir or not os.path.exists(data_dir):
                print(f"    警告: 数据目录不存在: {data_dir}")
                continue
            
            print(f"  处理文件: {filename} ({test_name})")
            
            # 解析结果文件
            samples = self.parse_result_file(result_file)
            failed_samples = [s for s in samples if s['is_failed']]
            
            if not failed_samples:
                print(f"    没有失败的比较")
                continue
            
            print(f"    找到 {len(failed_samples)} 个失败比较")
            
            # 重测失败比较
            updated_samples = {}
            
            for sample in failed_samples:
                print(f"    重测比较 {sample['comparison_id']}: {sample['video1'][:50]}... vs {sample['video2'][:50]}...")
                
                new_gpt_response = self.retry_failed_comparison(sample, data_dir, config)
                
                if new_gpt_response:
                    updated_samples[sample['comparison_id']] = new_gpt_response
                    total_successful += 1
                
                total_retried += 1
                
                # 添加延迟避免API限制
                time.sleep(15)
            
            # 更新结果文件
            if updated_samples:
                self.update_result_file(result_file, samples, updated_samples)
            
            print(f"    完成: {len(updated_samples)}/{len(failed_samples)} 个比较重测成功")
        
        print(f"  实验目录处理完成: 总计重测 {total_retried} 个比较，成功 {total_successful} 个")
    
    def dry_run_analysis(self, target_experiments: Optional[List[str]] = None) -> None:
        """
        试运行分析，统计失败比较数量但不进行重测
        
        参数:
            target_experiments: 目标实验列表，如果为None则处理所有实验
        """
        print("🔍 试运行模式：统计失败比较数量")
        
        # 查找所有实验目录
        all_experiments = list(self.experiment_configs.keys())
        
        if target_experiments:
            # 验证目标实验是否存在
            valid_experiments = []
            for exp in target_experiments:
                if exp in all_experiments and os.path.exists(exp):
                    valid_experiments.append(exp)
                else:
                    print(f"警告: 实验目录不存在或配置未知: {exp}")
            experiments_to_process = valid_experiments
        else:
            # 处理所有存在的实验目录
            experiments_to_process = [exp for exp in all_experiments if os.path.exists(exp)]
        
        if not experiments_to_process:
            print("❌ 没有找到可处理的实验目录")
            return
        
        print(f"分析 {len(experiments_to_process)} 个实验目录:")
        for exp in experiments_to_process:
            print(f"  - {exp}")
        print()
        
        # 统计总体情况
        total_experiments = 0
        total_files = 0
        total_comparisons = 0
        total_failed = 0
        experiment_stats = []
        
        # 分析每个实验目录
        for exp_dir in experiments_to_process:
            exp_name = os.path.basename(exp_dir)
            
            if exp_name not in self.experiment_configs:
                continue
            
            print(f"分析实验: {exp_name}")
            
            # 查找结果文件
            result_files = glob.glob(os.path.join(exp_dir, "*relative_results.txt"))
            
            if not result_files:
                print(f"  警告: 未找到结果文件")
                continue
            
            exp_total_comparisons = 0
            exp_total_failed = 0
            file_stats = []
            
            # 分析每个结果文件
            for result_file in result_files:
                filename = os.path.basename(result_file)
                test_name = self.get_test_name_from_filename(filename)
                
                # 解析结果文件
                samples = self.parse_result_file(result_file)
                failed_samples = [s for s in samples if s['is_failed']]
                
                file_stats.append({
                    'filename': filename,
                    'test_name': test_name,
                    'total_comparisons': len(samples),
                    'failed_comparisons': len(failed_samples),
                    'success_rate': (len(samples) - len(failed_samples)) / len(samples) * 100 if samples else 0
                })
                
                exp_total_comparisons += len(samples)
                exp_total_failed += len(failed_samples)
                
                print(f"  {test_name}: {len(failed_samples)}/{len(samples)} 失败 "
                      f"({(len(samples) - len(failed_samples)) / len(samples) * 100:.1f}% 成功率)")
            
            experiment_stats.append({
                'experiment': exp_name,
                'total_comparisons': exp_total_comparisons,
                'failed_comparisons': exp_total_failed,
                'success_rate': (exp_total_comparisons - exp_total_failed) / exp_total_comparisons * 100 if exp_total_comparisons else 0,
                'file_stats': file_stats
            })
            
            total_experiments += 1
            total_files += len(result_files)
            total_comparisons += exp_total_comparisons
            total_failed += exp_total_failed
            
            print(f"  实验汇总: {exp_total_failed}/{exp_total_comparisons} 失败 "
                  f"({(exp_total_comparisons - exp_total_failed) / exp_total_comparisons * 100:.1f}% 成功率)")
            print()
        
        # 打印汇总统计
        print("=" * 80)
        print("试运行统计汇总")
        print("=" * 80)
        print(f"总实验数: {total_experiments}")
        print(f"总文件数: {total_files}")
        print(f"总比较数: {total_comparisons}")
        print(f"失败比较数: {total_failed}")
        print(f"整体成功率: {(total_comparisons - total_failed) / total_comparisons * 100:.1f}%")
        print()
        
        # 按成功率排序显示实验
        experiment_stats.sort(key=lambda x: x['success_rate'])
        
        print("实验成功率排名 (从低到高):")
        print("-" * 80)
        for i, exp_stat in enumerate(experiment_stats, 1):
            print(f"{i:2d}. {exp_stat['experiment']:<60} "
                  f"{exp_stat['failed_comparisons']:3d}/{exp_stat['total_comparisons']:3d} 失败 "
                  f"({exp_stat['success_rate']:5.1f}% 成功)")
        
        print()
        print("需要重测的比较数量:")
        print("-" * 40)
        for exp_stat in experiment_stats:
            if exp_stat['failed_comparisons'] > 0:
                print(f"{exp_stat['experiment']:<60} {exp_stat['failed_comparisons']:3d} 个比较")
        
        print(f"\n总计需要重测: {total_failed} 个比较")
        print(f"预估重测时间: {total_failed * 15 / 60:.1f} 分钟 (按每比较15秒计算)")

    def run(self, target_experiments: Optional[List[str]] = None) -> None:
        """
        运行重测程序
        
        参数:
            target_experiments: 目标实验列表，如果为None则处理所有实验
        """
        print("开始重测失败的比较预测...")
        
        # 查找所有实验目录
        all_experiments = list(self.experiment_configs.keys())
        
        if target_experiments:
            # 验证目标实验是否存在
            valid_experiments = []
            for exp in target_experiments:
                if exp in all_experiments and os.path.exists(exp):
                    valid_experiments.append(exp)
                else:
                    print(f"警告: 实验目录不存在或配置未知: {exp}")
            experiments_to_process = valid_experiments
        else:
            # 处理所有存在的实验目录
            experiments_to_process = [exp for exp in all_experiments if os.path.exists(exp)]
        
        if not experiments_to_process:
            print("❌ 没有找到可处理的实验目录")
            return
        
        print(f"将处理 {len(experiments_to_process)} 个实验目录:")
        for exp in experiments_to_process:
            print(f"  - {exp}")
        
        # 处理每个实验目录
        total_start_time = time.time()
        
        for exp_dir in experiments_to_process:
            try:
                self.process_experiment_directory(exp_dir)
            except Exception as e:
                print(f"❌ 处理实验目录 {exp_dir} 时出错: {e}")
                continue
        
        total_time = time.time() - total_start_time
        print(f"\n✅ 重测程序完成，总耗时: {total_time:.1f} 秒")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="重测elasticity relative ablation实验中的失败预测")
    parser.add_argument(
        "--experiments", 
        nargs="*", 
        help="指定要处理的实验目录名称，如果不指定则处理所有实验"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="试运行模式，只统计失败比较数量，不实际重测"
    )
    
    args = parser.parse_args()
    
    # 创建重测处理器
    retrier = RetryFailedPredictionsRelative()
    
    if args.dry_run:
        # 运行试运行分析
        retrier.dry_run_analysis(target_experiments=args.experiments)
        return
    
    # 运行重测
    retrier.run(target_experiments=args.experiments)


if __name__ == "__main__":
    main() 