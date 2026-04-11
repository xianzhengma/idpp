import os
from openai import OpenAI

client = OpenAI(
    # If environment variables are not configured, replace the line below with: api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    # qwen-vl-max is used as an example here. For more models: https://www.alibabacloud.com/help/en/model-studio/models
    model="qwen-vl-max",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    "https://img.alicdn.com/imgextra/i3/O1CN01K3SgGo1eqmlUgeE9b_!!6000000003923-0-tps-3840-2160.jpg",
                    "https://img.alicdn.com/imgextra/i4/O1CN01BjZvwg1Y23CF5qIRB_!!6000000003000-0-tps-3840-2160.jpg",
                    "https://img.alicdn.com/imgextra/i4/O1CN01Ib0clU27vTgBdbVLQ_!!6000000007859-0-tps-3840-2160.jpg",
                    "https://img.alicdn.com/imgextra/i1/O1CN01aygPLW1s3EXCdSN4X_!!6000000005710-0-tps-3840-2160.jpg"]
            },
            {
                "type": "text",
                "text": "Describe this video"
            }]}]
)
print(completion.model_dump_json())