import json
import requests
import base64
import time
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_response(prompt, model_version=None, max_retries=5):
    """
    调用GPT API
    
    参数:
        prompt: 提示消息
        model_version: 模型版本
        max_retries: 最大重试次数
    
    返回:
        API响应内容或错误信息
    """
    key_path = os.environ.get(
        "GPT_KEY_JSON",
        os.path.join(os.environ.get("SVD_ROOT", "."), "secrets", "gpt_key.json"),
    )
    for idx in range(max_retries):
        try:
            with open(key_path, "r", encoding="utf-8") as file:
                gpt_key = json.load(file)
            key = gpt_key["key"] 
            model_version = gpt_key["version"] if model_version == None else model_version
            ip_address = gpt_key["ip_address"]
            header = {
                "Content-Type":"application/json",
                "Authorization": f"Bearer {key}"
            }
            post_dict = {
                "model": model_version,
                "messages": prompt,
            }
            r = requests.post(ip_address, json=post_dict, headers=header)
            json_r = r.json()
            message = json_r["choices"][0]["message"]["content"]
            return message
        except Exception as e:
            print(f"重试 {idx+1}/{max_retries} 失败: {e}")
            if hasattr(locals(), 'r'):
                print(f"HTTP状态码: {r.status_code}")
            if hasattr(locals(), 'json_r'):
                print(f"响应内容: {json_r}")
            
            if idx < max_retries - 1:
                wait_time = 30 * (idx + 1)  # 递增等待时间
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            
    # 所有重试都失败了
    return "GPT API调用失败: 所有重试都失败了，请检查API服务状态"

def gptv_response(prompt, model_version=None, max_retries=8, base_delay=30):
    """
    调用GPT Vision API（增强重试机制）
    
    参数:
        prompt: 提示消息
        model_version: 模型版本
        max_retries: 最大重试次数（增加到8次）
        base_delay: 基础延迟时间（秒）
    
    返回:
        API响应内容或错误信息
    """
    for idx in range(max_retries):
        try:
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                return "GPT Vision API 调用失败: 请设置环境变量 OPENAI_API_KEY"
            print("使用环境变量 OPENAI_API_KEY（已隐藏完整内容）")
            model_version = "gpt-4o" if model_version == None else model_version
            ip_address = "https://api.chatanywhere.tech/v1/chat/completions"  # 修正API endpoint
            header = {
                "Content-Type":"application/json",
                "Authorization": f"Bearer {key}", 
            }
            post_dict = {
                "model": model_version,
                "messages": prompt,
                "max_tokens": 2000,
                "temperature": 0.0
            }
            
            # 增加超时时间，并设置连接和读取超时
            r = requests.post(ip_address, json=post_dict, headers=header, timeout=(30, 120))
            print(f"<Response [{r.status_code}]>")
            
            # 检查HTTP状态码
            if r.status_code == 200:
                json_r = r.json()
                
                # 检查响应格式
                if "choices" in json_r and len(json_r["choices"]) > 0:
                    message = json_r["choices"][0]["message"]["content"]
                    print(f"✓ API调用成功")
                    return message
                else:
                    print(f"响应格式错误: {json_r}")
                    if idx < max_retries - 1:
                        wait_time = base_delay * (2 ** min(idx, 4))  # 指数退避，最大32倍base_delay
                        print(f"等待 {wait_time} 秒后重试 ({idx+1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
            
            elif r.status_code == 429:
                # 速率限制错误，特殊处理
                print(f"遇到速率限制 (429)")
                retry_after = r.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after) + 10  # 额外等待10秒
                else:
                    wait_time = base_delay * (2 ** min(idx, 4))  # 指数退避
                
                if idx < max_retries - 1:
                    print(f"等待 {wait_time} 秒后重试 ({idx+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
            
            elif r.status_code in [500, 502, 503, 504]:
                # 服务器错误，可以重试
                print(f"服务器错误: {r.status_code}")
                if idx < max_retries - 1:
                    wait_time = base_delay * (2 ** min(idx, 3))  # 服务器错误用较短的退避时间
                    print(f"等待 {wait_time} 秒后重试 ({idx+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
            
            else:
                # 其他HTTP错误
                print(f"HTTP错误: {r.status_code}")
                print(f"响应内容: {r.text[:200]}...")
                if idx < max_retries - 1:
                    wait_time = base_delay * (idx + 1)  # 线性增长
                    print(f"等待 {wait_time} 秒后重试 ({idx+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                    
        except requests.exceptions.Timeout as e:
            print(f"请求超时 ({idx+1}/{max_retries}): {e}")
            if idx < max_retries - 1:
                wait_time = base_delay * (idx + 1)
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
                
        except requests.exceptions.ConnectionError as e:
            print(f"连接错误 ({idx+1}/{max_retries}): {e}")
            if idx < max_retries - 1:
                wait_time = base_delay * (idx + 1)
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"请求异常 ({idx+1}/{max_retries}): {e}")
            if idx < max_retries - 1:
                wait_time = base_delay * (idx + 1)
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
                
        except json.JSONDecodeError as e:
            print(f"JSON解析错误 ({idx+1}/{max_retries}): {e}")
            if 'r' in locals():
                print(f"原始响应: {r.text[:200]}...")
            if idx < max_retries - 1:
                wait_time = base_delay * (idx + 1)
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
                
        except Exception as e:
            print(f"未知错误 ({idx+1}/{max_retries}): {e}")
            if 'r' in locals():
                print(f"HTTP状态码: {r.status_code}")
                print(f"响应内容: {r.text[:200]}...")
            if 'json_r' in locals():
                print(f"JSON响应: {json_r}")
            
            if idx < max_retries - 1:
                wait_time = base_delay * (idx + 1)
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            
    # 所有重试都失败了
    return "GPT Vision API调用失败: 所有重试都失败了，请检查API服务状态或稍后再试"

# 测试代码
def test_gptv_with_image():
    try:
        # 图像路径
        image_path = os.environ.get("GPT_SAMPLE_IMAGE", "cover.png")
        
        # 编码图像为 base64
        base64_image = encode_image(image_path)
        print("Image encoded successfully.")
        
        # 构造提示（prompt），包含文本和图像
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请描述这张图片的内容。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # 调用 gptv_response 函数
        print("Sending request to GPT-V model...")
        response = gptv_response(prompt)
        print("Response from GPT-V model:")
        print(response)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_gptv_with_image()