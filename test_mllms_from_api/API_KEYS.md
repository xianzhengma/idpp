# API keys (open-source release)

Do not commit real secrets. Use environment variables:

- `OPENAI_API_KEY` — OpenAI or compatible proxies (some shell scripts and `gpt_api_third_party.gptv_response`)
- `GPT_KEY_JSON` / `secrets/gpt_key.json` — JSON consumed by `gpt_response()` (see repo root `secrets/gpt_key.json.example`)
- `DASHSCOPE_API_KEY` — Qwen (DashScope)
- `GPT_PROXY_KEY` / `GPT_PROXY_URL` — legacy `gpt_zhan.py` experiments
