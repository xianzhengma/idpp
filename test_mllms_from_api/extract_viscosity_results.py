import os
import re
import json
from datetime import datetime

def extract_viscosity_data(markdown_file):
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        video_name_match = re.search('File name\\*\\*: (.+?)\\.mov', content)
        if not video_name_match:
            video_name_match = re.search('File name\\*\\*: (.+?)\\.mp4', content)
        if video_name_match:
            video_name = video_name_match.group(1)
        else:
            base_name = os.path.basename(markdown_file)
            video_name = base_name.split('_analysis_')[0]
        gt_range_match = re.search('Real viscosity range \\(cP\\)\\*\\*: (\\d+(?:\\.\\d+)?) - (\\d+(?:\\.\\d+)?)', content)
        if not gt_range_match:
            gt_range_match = re.search('True viscosity range:(\\d+(?:\\.\\d+)?) - (\\d+(?:\\.\\d+)?) cP', content)
        if gt_range_match:
            gt_min = float(gt_range_match.group(1))
            gt_max = float(gt_range_match.group(2))
            gt_range = [gt_min, gt_max]
        else:
            gt_range = None
        gpt_range_patterns = ['Estimated viscosity range: (\\d+(?:\\.\\d+)?) - (\\d+(?:\\.\\d+)?) cP', 'Estimated viscosity range: (\\d+(?:\\.\\d+)?) - (\\d+(?:\\.\\d+)?) cP', 'Estimated viscosity: (\\d+(?:\\.\\d+)?) - (\\d+(?:\\.\\d+)?) cP', 'viscosity: (\\d+(?:\\.\\d+)?) - (\\d+(?:\\.\\d+)?) cP', 'viscosity range: (\\d+(?:\\.\\d+)?) - (\\d+(?:\\.\\d+)?) cP', 'viscosity: (\\d+(?:\\.\\d+)?) - (\\d+(?:\\.\\d+)?) cP', 'estimated viscosity.*?(\\d+(?:\\.\\d+)?) - (\\d+(?:\\.\\d+)?) cP', 'estimated viscosity.*?(\\d+(?:\\.\\d+)?) to (\\d+(?:\\.\\d+)?) cP', '\\*\\*(\\d+(?:\\.\\d+)?) to (\\d+(?:\\.\\d+)?) cP\\*\\*', '\\*\\*(\\d+(?:\\.\\d+)?) - (\\d+(?:\\.\\d+)?) cP\\*\\*', '(\\d+(?:\\.\\d+)?) - (\\d+(?:\\.\\d+)?) cP']
        gpt_range = None
        for pattern in gpt_range_patterns:
            gpt_range_match = re.search(pattern, content, re.IGNORECASE)
            if gpt_range_match:
                gpt_min = float(gpt_range_match.group(1))
                gpt_max = float(gpt_range_match.group(2))
                gpt_range = [gpt_min, gpt_max]
                break
        if gpt_range is None:
            single_value_patterns = ['Estimated viscosity: (\\d+(?:\\.\\d+)?) cP', 'viscosity: (\\d+(?:\\.\\d+)?) cP', 'viscosity: (\\d+(?:\\.\\d+)?) cP', '\\*\\*(\\d+(?:\\.\\d+)?) cP\\*\\*']
            for pattern in single_value_patterns:
                single_value_match = re.search(pattern, content, re.IGNORECASE)
                if single_value_match:
                    value = float(single_value_match.group(1))
                    gpt_range = [value, value]
                    break
        return {'video_name': video_name, 'gt_range': gt_range, 'gpt_range': gpt_range}
    except Exception as e:
        print(f'Process files{markdown_file}An error occurred:{str(e)}')
        return {'video_name': os.path.basename(markdown_file).split('_analysis_')[0], 'gt_range': None, 'gpt_range': None, 'error': str(e)}

def main():
    markdown_folder = 'gpt/viscosity_results/test3_new'
    if not os.path.exists(markdown_folder):
        print(f'Error: folder{markdown_folder}does not exist')
        return
    markdown_files = [f for f in os.listdir(markdown_folder) if f.endswith('.md')]
    if not markdown_files:
        print(f'in folder{markdown_folder}Markdown file not found in')
        return
    print(f'in folder{markdown_folder}found in{len(markdown_files)}Markdown files')
    results = []
    success_count = 0
    failed_count = 0
    for md_file in markdown_files:
        file_path = os.path.join(markdown_folder, md_file)
        data = extract_viscosity_data(file_path)
        if data['gpt_range'] is None:
            print(f'Warning: Unable to retrieve file{md_file}Extract the GPT estimated viscosity range from')
            failed_count += 1
        else:
            success_count += 1
        results.append(data)
        print(f'Process files:{md_file}')
        print(f"Video name:{data['video_name']}")
        print(f"True viscosity range:{data['gt_range']}")
        print(f"GPT estimated viscosity range:{data['gpt_range']}")
        print()    result_dict = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'total_files': len(results), 'success_count': success_count, 'failed_count': failed_count, 'results': results}
    output_file = 'viscosity_analysis_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f'Results have been saved to:{output_file}')
    print(f'Successful extraction:{success_count}/{len(results)}document')
    print(f'Extraction failed:{failed_count}/{len(results)}document')
if __name__ == '__main__':
    main()
