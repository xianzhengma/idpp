from IPython.display import display, Image, Audio
import cv2
import base64
import time
from openai import OpenAI
import os
import requests
import sys
import numpy as np
from datetime import datetime
import json
import re

# 定义全局变量
output_directory = "gpt_analysis_results"  # 默认值

def read_video_frames(video_path, frame_step=5, frame_size=None):
    """
    读取视频文件并将帧转换为base64编码，每隔frame_step帧取一帧
    
    参数:
        video_path: 视频文件的路径
        frame_step: 每隔多少帧取一帧
        frame_size: 调整帧大小的元组 (宽度, 高度)，None表示保持原始大小
        
    返回:
        base64编码的视频帧列表
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
        
    base64_frames = []
    frame_count = 0
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        # 每隔frame_step帧取一帧
        if frame_count % frame_step == 0:
            # 如果指定了帧大小，则调整帧的大小
            if frame_size is not None:
                frame = cv2.resize(frame, frame_size)
                
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
            
        frame_count += 1

    video.release()
    
    if frame_size:
        print(f"总共 {frame_count} 帧，采样后得到 {len(base64_frames)} 帧，调整大小为 {frame_size}。")
    else:
        print(f"总共 {frame_count} 帧，采样后得到 {len(base64_frames)} 帧，保持原始大小。")
        
    return base64_frames

def display_frames(base64_frames, is_jupyter=False, delay=0.025):
    """
    显示视频帧
    
    参数:
        base64_frames: base64编码的视频帧列表
        is_jupyter: 是否在Jupyter环境中运行
        delay: 帧之间的延迟时间(秒)
    """
    if is_jupyter:
        # 在Jupyter环境中使用IPython的display功能
        display_handle = display(None, display_id=True)
        for img in base64_frames:
            display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
            time.sleep(delay)
    else:
        # 在非Jupyter环境中使用OpenCV显示
        print("在OpenCV窗口中显示视频...")
        for img in base64_frames:
            frame = cv2.imdecode(
                np.frombuffer(base64.b64decode(img.encode("utf-8")), np.uint8),
                cv2.IMREAD_COLOR
            )
            cv2.imshow("Video", frame)
            if cv2.waitKey(int(delay * 1000)) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

def analyze_video_with_gpt(video_path, model="gpt-4o", analysis_type="restitution", max_frames=16, frame_size=None, temperature=0, max_retries=5):
    """
    使用GPT模型分析视频帧
    
    参数:
        video_path: 视频文件路径
        model: 要使用的GPT模型名称
        analysis_type: 分析类型，"restitution"表示弹性系数，"viscosity"表示粘度
        max_frames: 最大帧数
        frame_size: 调整帧大小的元组 (宽度, 高度)，None表示保持原始大小
        temperature: 控制生成文本的随机性，0表示最确定性的输出
        max_retries: 遇到速率限制时的最大重试次数
        
    返回:
        GPT模型的分析结果
    """
    # 从环境变量获取API密钥
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未设置OPENAI_API_KEY环境变量")
        
    client = OpenAI(api_key=api_key)
    
    # 读取视频帧
    base64_frames = read_video_frames(video_path, frame_step=10, frame_size=frame_size)
    print(f"采样了 {len(base64_frames)} 帧用于分析")
    
    # 限制帧数以避免超出API限制
    if len(base64_frames) > max_frames:
        # 均匀选择帧
        indices = [int(i * (len(base64_frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
        selected_frames = [base64_frames[i] for i in indices]
        print(f"选择了 {len(selected_frames)} 帧发送到API")
    else:
        selected_frames = base64_frames
    
    # 根据分析类型选择不同的提示
    if analysis_type == "restitution":
        # 弹性系数分析提示
        prompt_text = """
        Please analyze these frames from a video of a bouncing object and estimate its restitution coefficient.
        
        The restitution coefficient is the ratio of the velocity after collision to the velocity before collision.
        
        Please:
        1. Track the object's position in each frame
        2. Determine its velocity before and after impact
        3. Calculate the restitution coefficient
        4. Provide a detailed analysis of your methodology
        
        YOUR RESPONSE MUST END WITH A SPECIFIC RESTITUTION COEFFICIENT VALUE IN THIS FORMAT:
        "Estimated restitution coefficient value: X"
        
        Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.
        """
    elif analysis_type == "viscosity":
        # 粘度分析提示 - 更直接的指令版本，针对单一值
        prompt_text = """
        You are analyzing frames from a video showing liquid motion. Your task is to estimate the viscosity of this liquid.
        
        IMPORTANT: You CAN analyze these images. You MUST provide a specific viscosity estimate.
        
        For reference:
        - Water: 1 cP
        - Vegetable oil: 100 cP
        - Motor oil: 300 cP
        - Honey: 5000 cP
        
        Analyze the liquid's flow speed, surface waves, and interaction with objects.
        
        YOUR RESPONSE MUST END WITH A SPECIFIC VISCOSITY VALUE IN THIS FORMAT:
        "Estimated viscosity value: X cP"
        
        Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.
        """
    else:
        raise ValueError(f"不支持的分析类型: {analysis_type}")
    
    # 构建消息内容
    prompt_messages = [
        {
            "role": "system",
            "content": "You are an expert in fluid dynamics with the ability to analyze video frames and determine liquid viscosity. You MUST provide a specific viscosity estimate in centipoise (cP). Never say you cannot analyze the images."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}} for frame in selected_frames],
            ],
        },
    ]
    
    # 发送请求，带有重试机制
    print("发送请求到GPT API...")
    
    for retry in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=prompt_messages,
                max_tokens=2000,
                temperature=temperature,  # 设置温度参数
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            print(f"错误详情: {error_msg}")
            
            # 检查是否是速率限制错误
            if "rate_limit_exceeded" in error_msg and retry < max_retries:
                # 尝试从错误消息中提取建议的等待时间
                wait_time_match = re.search(r"Please try again in (\d+\.\d+)s", error_msg)
                if wait_time_match:
                    wait_time = float(wait_time_match.group(1)) + 30  # 额外等待30秒以确保安全
                else:
                    wait_time = 30 * (retry + 1)  # 指数退避策略，从30秒开始
                
                print(f"遇到速率限制，等待 {wait_time:.2f} 秒后重试 ({retry+1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                # 其他错误或已达到最大重试次数
                if retry < max_retries:
                    # 对于其他错误也尝试重试
                    wait_time = 30 * (retry + 1)
                    print(f"遇到错误，等待 {wait_time:.2f} 秒后重试 ({retry+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"GPT API调用出错: {error_msg}\n\n建议:\n1. 检查您的OpenAI API密钥是否有效\n2. 确认是否有足够的API额度\n3. 考虑减少视频大小或长度"

def load_viscosity_gt_data(json_path):
    """
    从JSON文件加载粘度真实值数据
    
    参数:
        json_path: JSON文件路径
        
    返回:
        包含视频文件名和对应粘度值的字典
    """
    try:
        with open(json_path, 'r') as f:
            gt_data = json.load(f)
        print(f"成功加载了 {len(gt_data)} 条粘度真实值数据")
        return gt_data
    except Exception as e:
        print(f"加载粘度真实值数据时出错: {str(e)}")
        return {}

def save_result_to_markdown(video_path, analysis_result, model_name, analysis_type="restitution", viscosity_gt_data=None):
    """
    将分析结果保存为Markdown文件
    
    参数:
        video_path: 分析的视频路径
        analysis_result: GPT的分析结果
        model_name: 使用的模型名称
        analysis_type: 分析类型，"restitution"表示弹性系数，"viscosity"表示粘度
        viscosity_gt_data: 粘度真实值数据字典
    """
    global output_directory
    
    # 提取视频文件名（不含路径和扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建Markdown文件名
    markdown_file = f"{output_directory}/{video_name}_analysis_{timestamp}.md"
    
    # 根据分析类型设置标题和提取真实值（如果存在）
    if analysis_type == "restitution":
        title = "弹跳物体视频分析"
        # 提取视频中的恢复系数（如果存在于文件名中）
        true_value = None
        if "E_" in video_name:
            try:
                # 尝试从文件名中提取恢复系数
                parts = video_name.split("E_")[1].split("_")[0]
                true_value = float(parts)
            except:
                pass
        true_value_label = "真实恢复系数"
        true_value_display = f"{true_value}" if true_value is not None else "未知"
    elif analysis_type == "viscosity":
        title = "液体粘度视频分析"
        # 从JSON数据中获取粘度值
        true_value = None
        if viscosity_gt_data and video_name in viscosity_gt_data:
            true_value = viscosity_gt_data[video_name]
        true_value_label = "真实粘度值 (cP)"
        if true_value:
            # 检查true_value是否为列表（范围）或单一值
            if isinstance(true_value, list):
                true_value_display = f"{true_value[0]} - {true_value[1]}"
            else:
                true_value_display = f"{true_value}"
        else:
            # 尝试从文件名中提取粘度（如果存在）
            if "V_" in video_name:
                try:
                    parts = video_name.split("V_")[1].split("_")[0]
                    true_value_display = float(parts)
                except:
                    true_value_display = "未知"
            else:
                true_value_display = "未知"
    else:
        title = "视频分析"
        true_value = None
        true_value_label = "真实值"
        true_value_display = "未知"
    
    # 构建Markdown内容
    markdown_content = f"""# {title}

## 视频信息
- **文件名**: {os.path.basename(video_path)}
- **分析时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **使用模型**: {model_name}
- **分析类型**: {"弹性系数" if analysis_type == "restitution" else "粘度"}
- **{true_value_label}**: {true_value_display}

## GPT分析结果

{analysis_result}

"""
    
    # 写入Markdown文件
    with open(markdown_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"\nMarkdown分析报告已保存至: {markdown_file}")

def get_processed_videos(output_dir):
    """
    获取已经处理过的视频文件名列表
    
    参数:
        output_dir: 输出目录路径
        
    返回:
        已处理视频的基本文件名列表（不含扩展名）
    """
    if not os.path.exists(output_dir):
        return []
        
    processed_files = []
    
    # 查找所有分析结果文件
    for file in os.listdir(output_dir):
        if file.endswith('.md') and '_analysis_' in file:
            # 从文件名中提取视频名称
            # 格式: video_name_analysis_date.md
            video_name = file.split('_analysis_')[0]
            processed_files.append(video_name)
            
    return processed_files

def process_all_videos_in_folder(folder_path, model_name="gpt-4o", analysis_type="restitution", 
                                viscosity_gt_data=None, frame_size=None, temperature=0, 
                                processed_videos=None):
    """
    处理指定文件夹中的所有视频文件
    
    参数:
        folder_path: 包含视频文件的文件夹路径
        model_name: 要使用的GPT模型名称
        analysis_type: 分析类型，"restitution"表示弹性系数，"viscosity"表示粘度
        viscosity_gt_data: 粘度真实值数据字典
        frame_size: 调整帧大小的元组 (宽度, 高度)，None表示保持原始大小
        temperature: 控制生成文本的随机性，0表示最确定性的输出
        processed_videos: 已处理过的视频列表，如果提供则跳过这些视频
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在或不是一个目录")
        return
    
    # 获取文件夹中的所有文件
    all_files = os.listdir(folder_path)
    
    # 筛选出视频文件 (mp4, avi, mov等)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in video_extensions]
    
    if not video_files:
        print(f"在文件夹 {folder_path} 中没有找到视频文件")
        return
    
    # 如果提供了已处理视频列表，则过滤掉已处理的视频
    if processed_videos:
        unprocessed_files = []
        for video_file in video_files:
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            if video_name not in processed_videos:
                unprocessed_files.append(video_file)
        
        if not unprocessed_files:
            print(f"所有视频已经处理完成，没有新的视频需要处理")
            return
            
        print(f"在文件夹 {folder_path} 中找到 {len(video_files)} 个视频文件，其中 {len(unprocessed_files)} 个尚未处理")
        video_files = unprocessed_files
    else:
        print(f"在文件夹 {folder_path} 中找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频文件
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(folder_path, video_file)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"\n[{i+1}/{len(video_files)}] 处理视频: {video_file}")
        
        # 如果是粘度分析，显示真实粘度范围（如果有）
        if analysis_type == "viscosity" and viscosity_gt_data and video_name in viscosity_gt_data:
            viscosity_range = viscosity_gt_data[video_name]
            print(f"真实粘度范围: {viscosity_range[0]} - {viscosity_range[1]} cP")
        
        try:
            # 使用GPT分析视频
            analysis_result = analyze_video_with_gpt(video_path, model_name, analysis_type, frame_size=frame_size, temperature=temperature, max_retries=5)
            
            # 将结果保存为Markdown文件
            save_result_to_markdown(video_path, analysis_result, model_name, analysis_type, viscosity_gt_data)
            
            # 添加一些延迟以避免API限制
            if i < len(video_files) - 1:
                delay_time = 60  # 增加到60秒延迟，避免速率限制
                print(f"等待 {delay_time} 秒后处理下一个视频...")
                time.sleep(delay_time)
                
        except Exception as e:
            print(f"处理视频 {video_file} 时出错: {str(e)}")
            # 即使出错也等待一段时间再继续
            time.sleep(30)
            continue
    
    print("\n所有视频处理完成!")

def main():
    """
    主函数，处理视频分析流程
    """
    # 调试模式 - 设置为True只处理一个视频
    debug_mode = False
    
    # 选择分析类型
    # analysis_type = "viscosity"  # 可选值: "restitution" 或 "viscosity"
    analysis_type = "restitution"  # 可选值: "restitution" 或 "viscosity"

    
    # 设置帧大小 (宽度, 高度)
    # 常用分辨率: 
    # - None: 保持原始大小
    # - (640, 480): 标准清晰度
    # - (1280, 720): 高清
    # - (480, 360): 低分辨率，节省带宽
    frame_size = None  # 降低分辨率以减少令牌使用
    
    # 设置温度参数
    temperature = 0.1  # 0表示最确定性的输出，1表示最随机的输出
    
    # 根据分析类型设置文件夹和输出路径
    if analysis_type == "restitution":
        # 弹性系数分析
        # 定义多个测试文件夹及其对应的输出目录
        test_folders = [
            # {"folder": "bounce_ball_test1", "output": "gpt/bounce_ball_results/test1_new"},
            {"folder": "bounce_ball_test2", "output": "gpt/bounce_ball_results/test2_new"},
            {"folder": "bounce_ball_test3", "output": "gpt/bounce_ball_results/test3_new"}
        ]
    else:
        # 粘度分析
        test_folders = [
            {"folder": "viscosity_test1_merged_3.16_subset", "output": "gpt/viscosity_results/test1_new"},
            {"folder": "viscosity_test2_merged_3.16_subset", "output": "gpt/viscosity_results/test2_new"},
            {"folder": "viscosity_test3_merged_3.16_subset", "output": "gpt/viscosity_results/test3_new"}
        ]
    
    # 使用GPT模型
    model_name = "gpt-4o-mini"  # 可以使用 "gpt-4o-mini" 来降低成本
    
    # 批处理设置
    batch_size = 5  # 每次处理5个视频，然后暂停
    batch_pause = 60  # 批次之间暂停60秒
    
    # 循环处理每个测试文件夹
    for test_config in test_folders:
        folder_path = test_config["folder"]
        output_dir = test_config["output"]
        
        print(f"\n\n========== 开始处理文件夹: {folder_path} ==========\n")
        
        # 加载粘度真实值数据（如果是粘度分析）
        if analysis_type == "viscosity":
            gt_json_path = os.path.join(folder_path, "gt_retitution.json")
            if os.path.exists(gt_json_path):
                viscosity_gt_data = load_viscosity_gt_data(gt_json_path)
            else:
                print(f"警告: 未找到粘度真实值数据文件 {gt_json_path}")
                viscosity_gt_data = None
        else:
            viscosity_gt_data = None
        
        # 获取已处理的视频列表
        processed_videos = get_processed_videos(output_dir)
        print(f"已经处理过 {len(processed_videos)} 个视频")
        
        print(f"开始使用{model_name}分析文件夹中的视频...")
        print(f"分析类型: {'弹性系数' if analysis_type == 'restitution' else '粘度'}")
        print(f"分析结果将保存到: {output_dir}")
        print(f"调试模式: {'开启 - 只处理一个视频' if debug_mode else '关闭 - 批量处理视频'}")
        print(f"帧大小: {frame_size if frame_size else '原始大小'}")
        print(f"温度参数: {temperature}")
        print(f"批处理设置: 每批 {batch_size} 个视频，批次间隔 {batch_pause} 秒")
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 修改save_result_to_markdown函数中的输出目录
        global output_directory
        output_directory = output_dir
        
        if debug_mode:
            # 调试模式 - 只处理一个视频
            # 获取文件夹中的所有视频文件
            all_files = os.listdir(folder_path)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in video_extensions]
            
            if not video_files:
                print(f"在文件夹 {folder_path} 中没有找到视频文件")
                continue
            
            # 过滤已处理的视频
            unprocessed_files = []
            for video_file in video_files:
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                if video_name not in processed_videos:
                    unprocessed_files.append(video_file)
            
            if not unprocessed_files:
                print("所有视频已经处理完成，没有新的视频需要处理")
                continue
                
            # 选择第一个未处理的视频文件
            video_file = unprocessed_files[0]
            video_path = os.path.join(folder_path, video_file)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            print(f"\n调试模式 - 处理视频: {video_file}")
            
            # 如果是粘度分析，显示真实粘度范围（如果有）
            if analysis_type == "viscosity" and viscosity_gt_data and video_name in viscosity_gt_data:
                viscosity_range = viscosity_gt_data[video_name]
                print(f"真实粘度范围: {viscosity_range[0]} - {viscosity_range[1]} cP")
            
            try:
                # 使用GPT分析视频
                analysis_result = analyze_video_with_gpt(video_path, model_name, analysis_type, frame_size=frame_size, temperature=temperature)
                
                # 将结果保存为Markdown文件
                save_result_to_markdown(video_path, analysis_result, model_name, analysis_type, viscosity_gt_data)
                
            except Exception as e:
                print(f"处理视频 {video_file} 时出错: {str(e)}")
        else:
            # 正常模式 - 批量处理视频
            all_files = os.listdir(folder_path)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in video_extensions]
            
            # 过滤已处理的视频
            unprocessed_files = []
            for video_file in video_files:
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                if video_name not in processed_videos:
                    unprocessed_files.append(video_file)
            
            if not unprocessed_files:
                print("所有视频已经处理完成，没有新的视频需要处理")
                continue
                
            print(f"找到 {len(video_files)} 个视频文件，其中 {len(unprocessed_files)} 个尚未处理")
            video_files = unprocessed_files
            
            if not video_files:
                print(f"在文件夹 {folder_path} 中没有找到未处理的视频文件")
                continue
                
            # 按批次处理视频
            for i in range(0, len(video_files), batch_size):
                batch = video_files[i:i+batch_size]
                print(f"\n处理批次 {i//batch_size + 1}/{(len(video_files) + batch_size - 1)//batch_size}，包含 {len(batch)} 个视频")
                
                # 创建临时文件夹路径
                temp_folder = os.path.join(folder_path, "temp_batch")
                os.makedirs(temp_folder, exist_ok=True)
                
                # 复制当前批次的视频到临时文件夹
                for video_file in batch:
                    src_path = os.path.join(folder_path, video_file)
                    dst_path = os.path.join(temp_folder, video_file)
                    # 使用符号链接或复制文件
                    try:
                        os.symlink(os.path.abspath(src_path), dst_path)
                    except:
                        import shutil
                        shutil.copy2(src_path, dst_path)
                
                # 处理当前批次
                process_all_videos_in_folder(temp_folder, model_name, analysis_type, viscosity_gt_data, frame_size=frame_size, temperature=temperature)
                
                # 清理临时文件夹
                for file in os.listdir(temp_folder):
                    os.remove(os.path.join(temp_folder, file))
                os.rmdir(temp_folder)
                
                # 如果不是最后一个批次，则暂停
                if i + batch_size < len(video_files):
                    print(f"\n批次处理完成，暂停 {batch_pause} 秒后继续下一批次...")
                    time.sleep(batch_pause)

        print(f"\n========== 文件夹 {folder_path} 处理完成 ==========\n")
        
        # 在处理下一个文件夹之前暂停一段时间
        if test_config != test_folders[-1]:
            folder_pause = 120  # 文件夹之间暂停120秒
            print(f"暂停 {folder_pause} 秒后处理下一个文件夹...")
            time.sleep(folder_pause)

    print("\n所有文件夹处理完成!")

if __name__ == "__main__":
    main()