import os
import re
import json
from datetime import datetime

def extract_viscosity_data(markdown_file):
    """
    从Markdown文件中提取真实粘度范围和GPT估计的粘度范围
    
    参数:
        markdown_file: Markdown文件的路径
        
    返回:
        包含视频名称、真实粘度范围和GPT估计粘度范围的字典
    """
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取视频名称
        video_name_match = re.search(r'文件名\*\*: (.+?)\.mov', content)
        if not video_name_match:
            video_name_match = re.search(r'文件名\*\*: (.+?)\.mp4', content)
        
        if video_name_match:
            video_name = video_name_match.group(1)
        else:
            # 从文件名中提取视频名称
            base_name = os.path.basename(markdown_file)
            video_name = base_name.split('_analysis_')[0]
        
        # 提取真实粘度范围
        gt_range_match = re.search(r'真实粘度范围 \(cP\)\*\*: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?)', content)
        if not gt_range_match:
            gt_range_match = re.search(r'真实粘度范围: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?) cP', content)
        
        if gt_range_match:
            gt_min = float(gt_range_match.group(1))
            gt_max = float(gt_range_match.group(2))
            gt_range = [gt_min, gt_max]
        else:
            gt_range = None
        
        # 提取GPT估计的粘度范围
        # 尝试多种可能的格式
        gpt_range_patterns = [
            r'估计粘度范围: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?) cP',
            r'粘度范围: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?) cP',
            r'粘度估计: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?) cP',
            r'粘度: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?) cP',
            r'viscosity range: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?) cP',
            r'viscosity: (\d+(?:\.\d+)?) - (\d+(?:\.\d+)?) cP',
            r'estimated viscosity.*?(\d+(?:\.\d+)?) - (\d+(?:\.\d+)?) cP',
            r'estimated viscosity.*?(\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) cP',
            r'\*\*(\d+(?:\.\d+)?) to (\d+(?:\.\d+)?) cP\*\*',
            r'\*\*(\d+(?:\.\d+)?) - (\d+(?:\.\d+)?) cP\*\*',
            r'(\d+(?:\.\d+)?) - (\d+(?:\.\d+)?) cP'
        ]
        
        gpt_range = None
        for pattern in gpt_range_patterns:
            gpt_range_match = re.search(pattern, content, re.IGNORECASE)
            if gpt_range_match:
                gpt_min = float(gpt_range_match.group(1))
                gpt_max = float(gpt_range_match.group(2))
                gpt_range = [gpt_min, gpt_max]
                break
        
        # 如果没有找到范围，尝试查找单一值
        if gpt_range is None:
            single_value_patterns = [
                r'估计粘度: (\d+(?:\.\d+)?) cP',
                r'粘度: (\d+(?:\.\d+)?) cP',
                r'viscosity: (\d+(?:\.\d+)?) cP',
                r'\*\*(\d+(?:\.\d+)?) cP\*\*'
            ]
            
            for pattern in single_value_patterns:
                single_value_match = re.search(pattern, content, re.IGNORECASE)
                if single_value_match:
                    value = float(single_value_match.group(1))
                    gpt_range = [value, value]
                    break
        
        return {
            "video_name": video_name,
            "gt_range": gt_range,
            "gpt_range": gpt_range
        }
    
    except Exception as e:
        print(f"处理文件 {markdown_file} 时出错: {str(e)}")
        return {
            "video_name": os.path.basename(markdown_file).split('_analysis_')[0],
            "gt_range": None,
            "gpt_range": None,
            "error": str(e)
        }

def main():
    # 设置Markdown文件夹路径
    markdown_folder = "gpt/viscosity_results/test3_new"
    
    # 检查文件夹是否存在
    if not os.path.exists(markdown_folder):
        print(f"错误: 文件夹 {markdown_folder} 不存在")
        return
    
    # 获取所有Markdown文件
    markdown_files = [f for f in os.listdir(markdown_folder) if f.endswith('.md')]
    
    if not markdown_files:
        print(f"在文件夹 {markdown_folder} 中没有找到Markdown文件")
        return
    
    print(f"在文件夹 {markdown_folder} 中找到 {len(markdown_files)} 个Markdown文件")
    
    # 提取所有文件的粘度数据
    results = []
    success_count = 0
    failed_count = 0
    
    for md_file in markdown_files:
        file_path = os.path.join(markdown_folder, md_file)
        data = extract_viscosity_data(file_path)
        
        # 检查是否成功提取了GPT估计的粘度范围
        if data['gpt_range'] is None:
            print(f"警告: 无法从文件 {md_file} 中提取GPT估计的粘度范围")
            failed_count += 1
        else:
            success_count += 1
            
        results.append(data)
        print(f"处理文件: {md_file}")
        print(f"  视频名称: {data['video_name']}")
        print(f"  真实粘度范围: {data['gt_range']}")
        print(f"  GPT估计粘度范围: {data['gpt_range']}")
        print()
    
    # 创建结果字典
    result_dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(results),
        "success_count": success_count,
        "failed_count": failed_count,
        "results": results
    }
    
    # 保存为JSON文件
    output_file = "viscosity_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")
    print(f"成功提取: {success_count}/{len(results)} 文件")
    print(f"提取失败: {failed_count}/{len(results)} 文件")

if __name__ == "__main__":
    main() 