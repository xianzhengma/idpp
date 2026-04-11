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
output_directory = 'gpt_analysis_results'

def read_video_frames(video_path, frame_step=5, frame_size=None):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f'Unable to open video file:{video_path}')
    base64_frames = []
    frame_count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_step == 0:
            if frame_size is not None:
                frame = cv2.resize(frame, frame_size)
            _, buffer = cv2.imencode('.jpg', frame)
            base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
        frame_count += 1
    video.release()
    if frame_size:
        print(f'total{frame_count}Frame, obtained after sampling{len(base64_frames)}frame, resized to{frame_size}。')
    else:
        print(f'total{frame_count}Frame, obtained after sampling{len(base64_frames)}Frame, keeping original size.')
    return base64_frames

def display_frames(base64_frames, is_jupyter=False, delay=0.025):
    if is_jupyter:
        display_handle = display(None, display_id=True)
        for img in base64_frames:
            display_handle.update(Image(data=base64.b64decode(img.encode('utf-8'))))
            time.sleep(delay)
    else:
        print('Display video in OpenCV window...')
        for img in base64_frames:
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(img.encode('utf-8')), np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow('Video', frame)
            if cv2.waitKey(int(delay * 1000)) & 255 == ord('q'):
                break
        cv2.destroyAllWindows()

def analyze_video_with_gpt(video_path, model='gpt-4o', analysis_type='restitution', max_frames=16, frame_size=None, temperature=0, max_retries=5):
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY environment variable not set')
    client = OpenAI(api_key=api_key)
    base64_frames = read_video_frames(video_path, frame_step=10, frame_size=frame_size)
    print(f'Sampled{len(base64_frames)}frames used for analysis')
    if len(base64_frames) > max_frames:
        indices = [int(i * (len(base64_frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
        selected_frames = [base64_frames[i] for i in indices]
        print(f'selected{len(selected_frames)}Frame sent to API')
    else:
        selected_frames = base64_frames
    if analysis_type == 'restitution':
        prompt_text = '\n        Please analyze these frames from a video of a bouncing object and estimate its restitution coefficient.\n        \n        The restitution coefficient is the ratio of the velocity after collision to the velocity before collision.\n        \n        Please:\n        1. Track the object\'s position in each frame\n        2. Determine its velocity before and after impact\n        3. Calculate the restitution coefficient\n        4. Provide a detailed analysis of your methodology\n        \n        YOUR RESPONSE MUST END WITH A SPECIFIC RESTITUTION COEFFICIENT VALUE IN THIS FORMAT:\n        "Estimated restitution coefficient value: X"\n        \n        Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.\n        '
    elif analysis_type == 'viscosity':
prompt_text = '\n You are analyzing frames from a video showing liquid motion. Your task is to estimate the viscosity of this liquid.\n \n IMPORTANT: You CAN analyze these images. You MUST provide a specific viscosity estimate.\n \n For reference:\n - Water: 1 cP\n - Vegetable oil: 100 cP\n - Motor oil: 300 cP\n - Honey: 5000 cP\n \n Analyze the liquid\'s flow speed, surface waves, and interaction with objects.\n \n YOUR RESPONSE MUST END WITH A SPECIFIC VISCOSITY VALUE IN THIS FORMAT:\n "Estimated viscosity value: X cP"\n \n Do not say you cannot analyze the images. You must provide your best estimate based on the visual information.\n '
    else:
        raise ValueError(f'Unsupported analysis types:{analysis_type}')
    prompt_messages = [{'role': 'system', 'content': 'You are an expert in fluid dynamics with the ability to analyze video frames and determine liquid viscosity. You MUST provide a specific viscosity estimate in centipoise (cP). Never say you cannot analyze the images.'}, {'role': 'user', 'content': [{'type': 'text', 'text': prompt_text}, *[{'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{frame}'}} for frame in selected_frames]]}]
    print('Send request to GPT API...')
    for retry in range(max_retries + 1):
        try:
            response = client.chat.completions.create(model=model, messages=prompt_messages, max_tokens=2000, temperature=temperature)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(f'Error details:{error_msg}')
            if 'rate_limit_exceeded' in error_msg and retry < max_retries:
                wait_time_match = re.search('Please try again in (\\d+\\.\\d+)s', error_msg)
                if wait_time_match:
                    wait_time = float(wait_time_match.group(1)) + 30
                else:
                    wait_time = 30 * (retry + 1)
                print(f'Rate limit encountered, wait{wait_time:.2f}Try again in seconds ({retry + 1}/{max_retries})...')
                time.sleep(wait_time)
                continue
            elif retry < max_retries:
                wait_time = 30 * (retry + 1)
                print(f'Error encountered, wait{wait_time:.2f}Try again in seconds ({retry + 1}/{max_retries})...')
                time.sleep(wait_time)
                continue
            else:
                return f'GPT API call error:{error_msg}\n\nSuggestions:\n1. Check that your OpenAI API key yesno is valid\n2. Confirm yesno has sufficient API credit\n3. Consider reducing the video size or length'

def load_viscosity_gt_data(json_path):
    try:
        with open(json_path, 'r') as f:
            gt_data = json.load(f)
        print(f'Loaded successfully{len(gt_data)}Strip viscosity real value data')
        return gt_data
    except Exception as e:
        print(f'Error loading actual viscosity data:{str(e)}')
        return {}

def save_result_to_markdown(video_path, analysis_result, model_name, analysis_type='restitution', viscosity_gt_data=None):
    global output_directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    markdown_file = f'{output_directory}/{video_name}_analysis_{timestamp}.md'
    if analysis_type == 'restitution':
        title = 'Video analysis of bouncing objects'
        true_value = None
        if 'E_' in video_name:
            try:
                parts = video_name.split('E_')[1].split('_')[0]
                true_value = float(parts)
            except:
                pass
        true_value_label = 'true recovery coefficient'
        true_value_display = f'{true_value}' if true_value is not None else 'unknown'
    elif analysis_type == 'viscosity':
        title = 'Liquid viscosity video analysis'
        true_value = None
        if viscosity_gt_data and video_name in viscosity_gt_data:
            true_value = viscosity_gt_data[video_name]
        true_value_label = 'True viscosity value (cP)'
        if true_value:
            if isinstance(true_value, list):true_value_display = f'{true_value[0]} - {true_value[1]}'
            else:
                true_value_display = f'{true_value}'
        elif 'V_' in video_name:
            try:
                parts = video_name.split('V_')[1].split('_')[0]
                true_value_display = float(parts)
            except:
                true_value_display = 'unknown'
        else:
            true_value_display = 'unknown'
    else:
        title = 'Video analysis'
        true_value = None
        true_value_label = 'true value'
        true_value_display = 'unknown'
    markdown_content = f"# {title}\n\n## Video information\n- **File name**: {os.path.basename(video_path)}\n- **Analysis time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n- **Use model**: {model_name}\n- **Analysis type**: {('elastic coefficient' if analysis_type == 'restitution' else 'viscosity')}\n- **{true_value_label}**: {true_value_display}\n\n## GPT analysis results\n\n{analysis_result}\n\n"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print(f'\nMarkdown analysis report has been saved to: {markdown_file}')

def get_processed_videos(output_dir):
    if not os.path.exists(output_dir):
        return []
    processed_files = []
    for file in os.listdir(output_dir):
        if file.endswith('.md') and '_analysis_' in file:
            video_name = file.split('_analysis_')[0]
            processed_files.append(video_name)
    return processed_files

def process_all_videos_in_folder(folder_path, model_name='gpt-4o', analysis_type='restitution', viscosity_gt_data=None, frame_size=None, temperature=0, processed_videos=None):
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f'Error: folder{folder_path}Does not exist or is not a directory')
        return
    all_files = os.listdir(folder_path)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in video_extensions]
    if not video_files:
        print(f'in folder{folder_path}Video file not found in')
        return
    if processed_videos:
        unprocessed_files = []
        for video_file in video_files:
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            if video_name not in processed_videos:
                unprocessed_files.append(video_file)
        if not unprocessed_files:
            print(f'All videos have been processed, no new videos need to be processed')
            return
        print(f'in folder{folder_path}found in{len(video_files)}video files, including{len(unprocessed_files)}have not yet been processed')
        video_files = unprocessed_files
    else:
        print(f'in folder{folder_path}found in{len(video_files)}video files')
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(folder_path, video_file)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f'\n[{i + 1}/{len(video_files)}] Processing video:{video_file}')
        if analysis_type == 'viscosity' and viscosity_gt_data and (video_name in viscosity_gt_data):
            viscosity_range = viscosity_gt_data[video_name]
            print(f'True viscosity range:{viscosity_range[0]} - {viscosity_range[1]} cP')
        try:
            analysis_result = analyze_video_with_gpt(video_path, model_name, analysis_type, frame_size=frame_size, temperature=temperature, max_retries=5)
            save_result_to_markdown(video_path, analysis_result, model_name, analysis_type, viscosity_gt_data)
            if i < len(video_files) - 1:
                delay_time = 60
                print(f'wait{delay_time}Process the next video in seconds...')
                time.sleep(delay_time)
        except Exception as e:
            print(f'Process video{video_file}An error occurred:{str(e)}')
            time.sleep(30)
            continue
    print('\nAll video processing completed!')

def main():
    debug_mode = False
    analysis_type = 'restitution'
    frame_size = None
    temperature=0.1
    if analysis_type == 'restitution':test_folders = [{'folder': 'bounce_ball_test2', 'output': 'gpt/bounce_ball_results/test2_new'}, {'folder': 'bounce_ball_test3', 'output': 'gpt/bounce_ball_results/test3_new'}]
    else:
        test_folders = [{'folder': 'viscosity_test1_merged_3.16_subset', 'output': 'gpt/viscosity_results/test1_new'}, {'folder': 'viscosity_test2_merged_3.16_subset', 'output': 'gpt/viscosity_results/test2_new'}, {'folder': 'viscosity_test3_merged_3.16_subset', 'output': 'gpt/viscosity_results/test3_new'}]
    model_name = 'gpt-4o-mini'
    batch_size = 5
    batch_pause = 60
    for test_config in test_folders:
        folder_path = test_config['folder']
        output_dir = test_config['output']
        print(f'\n\n========== Start processing folder: {folder_path} ==========\n')
        if analysis_type == 'viscosity':
            gt_json_path = os.path.join(folder_path, 'gt_retitution.json')
            if os.path.exists(gt_json_path):
                viscosity_gt_data = load_viscosity_gt_data(gt_json_path)
            else:
                print(f'Warning: Viscosity true value data file not found{gt_json_path}')
                viscosity_gt_data = None
        else:
            viscosity_gt_data = None
        processed_videos = get_processed_videos(output_dir)
        print(f'Already processed{len(processed_videos)}videos')
        print(f'Get started{model_name}Analyze videos in folder...')
        print(f"Analysis type:{('elastic coefficient' if analysis_type == 'restitution' else 'viscosity')}")
        print(f'Analysis results will be saved to:{output_dir}')
        print(f"Debug mode:{('On - only processes one video' if debug_mode else 'Off - batch processing of videos')}")
        print(f"Frame size:{(frame_size if frame_size else 'original size')}")
        print(f'Temperature parameters:{temperature}')
        print(f'Batch settings: per batch{batch_size}videos, batch interval{batch_pause}Second')
        os.makedirs(output_dir, exist_ok=True)
        global output_directory
        output_directory = output_dir
        if debug_mode:
            all_files = os.listdir(folder_path)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in video_extensions]
            if not video_files:
                print(f'in folder{folder_path}Video file not found in')
                continue
            unprocessed_files = []
            for video_file in video_files:
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                if video_name not in processed_videos:
                    unprocessed_files.append(video_file)
            if not unprocessed_files:
                print('All videos have been processed, no new videos need to be processed')
                continue
            video_file = unprocessed_files[0]
            video_path = os.path.join(folder_path, video_file)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f'\nDebug mode - processing video:{video_file}')
            if analysis_type == 'viscosity' and viscosity_gt_data and (video_name in viscosity_gt_data):
                viscosity_range = viscosity_gt_data[video_name]
                print(f'True viscosity range:{viscosity_range[0]} - {viscosity_range[1]} cP')
            try:
                analysis_result = analyze_video_with_gpt(video_path, model_name, analysis_type, frame_size=frame_size, temperature=temperature)
                save_result_to_markdown(video_path, analysis_result, model_name, analysis_type, viscosity_gt_data)
            except Exception as e:
                print(f'Process video{video_file}An error occurred:{str(e)}')
        else:
            all_files = os.listdir(folder_path)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in video_extensions]
            unprocessed_files = []
            for video_file in video_files:
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                if video_name not in processed_videos:unprocessed_files.append(video_file)
            if not unprocessed_files:
                print('All videos have been processed, no new videos need to be processed')
                continue
            print(f'turn up{len(video_files)}video files, including{len(unprocessed_files)}have not yet been processed')
            video_files = unprocessed_files
            if not video_files:
                print(f'in folder{folder_path}No unprocessed video files found in')
                continue
            for i in range(0, len(video_files), batch_size):
                batch = video_files[i:i + batch_size]
                print(f'\nProcess batch {i // batch_size + 1}/{(len(video_files) + batch_size - 1) // batch_size},Include{len(batch)}videos')
                temp_folder = os.path.join(folder_path, 'temp_batch')
                os.makedirs(temp_folder, exist_ok=True)
                for video_file in batch:
                    src_path = os.path.join(folder_path, video_file)
                    dst_path = os.path.join(temp_folder, video_file)
                    try:
                        os.symlink(os.path.abspath(src_path), dst_path)
                    except:
                        import shutil
                        shutil.copy2(src_path, dst_path)
                process_all_videos_in_folder(temp_folder, model_name, analysis_type, viscosity_gt_data, frame_size=frame_size, temperature=temperature)
                for file in os.listdir(temp_folder):
                    os.remove(os.path.join(temp_folder, file))
                os.rmdir(temp_folder)
                if i + batch_size < len(video_files):
                    print(f'\nBatch processing completed, pause{batch_pause}Continue to next batch in seconds...')
                    time.sleep(batch_pause)
        print(f'\n==========document folder {folder_path} processing completed ==========\n')
        if test_config != test_folders[-1]:
            folder_pause = 120
            print(f'pause{folder_pause}Process next folder in seconds...')
            time.sleep(folder_pause)
    print('\nAll folders processed!')
if __name__ == '__main__':
    main()