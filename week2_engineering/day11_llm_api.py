import os
import json



# ⚠️ 把你的 API key 填在这里（或者用环境变量 DEEPSEEK_API_KEY）
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

if "填" in API_KEY:
    raise SystemExit("请先注册 deepseek.com 拿到 API key，填到本文件第 33 行！")

# ============================================================
# Part 1 · 最原始的调用：一次 HTTP POST 请求
# ============================================================
# 剥掉所有 SDK 包装，API 的本质就是：
#   向一个网址 POST 一段 JSON，收回来一段 JSON。
# 看懂这一段，任何"模型服务"对你都不再神秘。

print("=" * 60)
print("Part 1 · 原始 HTTP 请求")

import requests

url = f"{BASE_URL}/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL,
     "messages": [
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "1+1=？。"},
    ],
    "temperature": 0.7,
    "max_tokens": 100,
}
response = requests.post(url, headers=headers, json=payload, timeout=60)
print(f"  HTTP 状态码: {response.status_code}")   # 200 = 成功

data = response.json()
print(f"  返回的 JSON 顶层键: {list(data.keys())}")
print(f"  模型回复: {data['choices'][0]['message']['content']}")
print(f"  本次消耗: {data['usage']}")   # 看 prompt_tokens / completion_tokens

# ✅ 检查点：能指出回复文本藏在 data['choices'][0]['message']['content']


# ============================================================
# Part 2 · 官方 SDK + 参数实验
# ============================================================
# openai SDK 是行业事实标准，DeepSeek 完全兼容——只要改 base_url。
# 以后换任何兼容服务（硅基流动、通义、Kimi）都只改这两行。

print("=" * 60)
print("Part 2 · SDK 调用 + 参数实验")

from openai import OpenAI

client = OpenAI(api_key=API_KEY,base_url=BASE_URL)

def chat(message,temperature=0.7,max_tokens=200):
    resp = client.chat.completions.create(
        model = MODEL,
        messages=message,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# --- 实验 A：system prompt 决定"人设" ---
print("\n--- 实验 A：同一个问题，不同的 system ---")
q = "评价一下电影《肖申克的救赎》"
print("  [助理人设]  ", chat([{"role": "system", "content": "你是一个简明扼要的影评人。"},
                              {"role": "user", "content": q}], max_tokens=120))
print("  [小学生人设]", chat([{"role": "system", "content": "你是一个8岁小孩，用最简单的话回答。"},
                              {"role": "user", "content": q}], max_tokens=120))

# --- 实验 B：temperature 控制"随机性" ---
print("\n--- 实验 B：temperature 对比（同样的问题问 3 次）---")
for temp in [0.0, 1.5]:
    print(f"  temperature={temp}:")
    for i in range(3):
        out = chat([{"role": "user", "content": "给一部电影起个创意名字"}],
                   temperature=temp, max_tokens=30)
        print(f"    第{i+1}次: {out}")
# 0.0 时三次几乎一样（确定性，适合抽取/分类任务）
# 1.5 时三次五花八门（发散，适合写文案）——面试常考的参数

# ============================================================
# Part 3 · 流式输出（打字机效果）
# ============================================================
# ChatGPT 那种"一边生成一边显示"就是 stream=True。
# 原理：服务器每生成几个 token 就推送一小段，客户端逐段接收。

print("=" * 60)
print("Part 3 · 流式输出")

stream = client.chat.completions.create(
    model = MODEL,
    messages = [{"role": "user","content":"用三句话介绍注意力机制"}],
    stream = True
)
for chunk in stream:
    piece = chunk.choices[0].delta.content   # 每次只含几个字，可能为 None
    if piece:
        print(piece, end="", flush=True)     # flush=True 强制立即显示
print("\n")

# ============================================================
# Part 4 · 结构化输出：让模型只吐合法 JSON
# ============================================================
# 做 RAG/Agent 时，程序需要"能用代码解析"的结果，而不是一段散文。
# 两板斧：① response_format 强制 JSON ② prompt 里给出字段格式

print("=" * 60)
print("Part 4 · JSON 结构化输出 + 情感分析实战")

review = "The plot was thin and the ending felt rushed, but the lead actor was phenomenal."

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content":
         "你是影评分析器。对用户给的影评输出 JSON，格式："
         '{"sentiment": "positive|negative|mixed", "score": 0到10的数字, "keywords": ["关键词"]}。'
         "只输出 JSON，不要任何其他文字。"},
        {"role": "user", "content": review},
    ],
    response_format={"type": "json_object"},   # 强制 JSON mode
    temperature=0.0,                            # 抽取任务用 0，保证稳定
)

raw = resp.choices[0].message.content
result = json.loads(raw)     # 现在可以用代码处理了！
print(f"  原始返回: {raw}")
print(f"  解析后:   {result}")
print(f"  情感={result['sentiment']} 评分={result['score']} 关键词={result['keywords']}")

# ✅ 检查点：json.loads 能成功、能按 key 取值——这就是"结构化"的意义


# ============================================================
# Part 5 · 思考题（不用写代码，想清楚写进笔记）
# ============================================================
# 1. Day 10 微调的 BERT 和今天的 API 调用，各适合什么场景？
#    提示：成本(一次性训练 vs 按token付费)、数据需求、能力上限
# 2. temperature 该设多少：情感分类？写广告文案？代码补全？
# 3. 为什么 Part 4 要 temperature=0 而 Part 2 实验 B 要 1.5？
print("=" * 60)
print("Day 11 核心任务完成！")