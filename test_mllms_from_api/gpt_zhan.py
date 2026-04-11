"""
遗留实验脚本：已去除硬编码密钥。请改用环境变量 `GPT_PROXY_KEY`（Bearer token）
与 `GPT_PROXY_URL`（chat completions 端点），或直接使用 `gpt_api_third_party.py`。
"""

import json
import os
import requests
from tqdm import tqdm

filename = "flickr_test_captions_list_5.10.json"
with open(filename, "r", encoding="utf-8") as handle:
    data = json.load(handle)


def gpt_response(prompt, model_version=None):
    """
    输入: prompt — OpenAI 格式 messages；model_version — 模型名。
    输出: API 返回的文本内容。
    作用: 通过第三方代理调用 GPT（密钥来自环境变量，不落盘）。
    """
    proxy_key = os.environ.get("GPT_PROXY_KEY", "")
    proxy_url = os.environ.get(
        "GPT_PROXY_URL", "https://api.chatanywhere.tech/v1/chat/completions"
    )
    if not proxy_key:
        raise RuntimeError("请设置环境变量 GPT_PROXY_KEY")
    for idx in range(99999):
        try:
            model_version = model_version or "gpt-4o"
            header = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {proxy_key}",
            }
            post_dict = {
                "model": model_version,
                "messages": prompt,
                "n": 5,
                "temperature": 1,
            }
            response = requests.post(proxy_url, json=post_dict, headers=header, timeout=120)
            json_r = response.json()["choices"]
            return json_r
        except (requests.RequestException, KeyError, ValueError) as exc:
            print(f"重试 {idx + 1}: {exc}")
            if idx > 100:
                raise


# 以下逻辑保留结构；运行前需自备 filename 与有效密钥环境变量
if __name__ == "__main__":
    print("Configure GPT_PROXY_KEY and data file before running.")
