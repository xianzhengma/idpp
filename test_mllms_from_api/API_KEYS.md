# API 密钥（开源版）

请勿将真实密钥提交到 Git。请使用环境变量：

- `OPENAI_API_KEY` — OpenAI 或兼容代理（部分 shell 脚本与 `gpt_api_third_party.gptv_response`）
- `GPT_KEY_JSON` / `secrets/gpt_key.json` — `gpt_response()` 使用的 JSON（见仓库根 `secrets/gpt_key.json.example`）
- `DASHSCOPE_API_KEY` — 通义千问（DashScope）
- `GPT_PROXY_KEY` / `GPT_PROXY_URL` — 遗留 `gpt_zhan.py` 实验入口
