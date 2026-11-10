# -*- coding: utf-8 -*-
"""
Day 12 · Function Calling + Prompt 工程
=======================================
上半天：Prompt 工程——同一句话怎么说，模型才能答得好
下半天：Function Calling——让模型学会"用工具"

Function Calling 核心认知（今天最重要的一句话）：
    模型不会执行任何函数！它只是输出"我想调用XX函数+参数"，
    真正执行的是你的代码，执行结果回传给模型，它再据此回答。
    模型 = 出主意的，你的代码 = 干活的。

准备工作：延续 Day 11 的 DeepSeek key（填到第 30 行）

术语提前知道：
  few-shot   = 在 prompt 里给几个例子
  CoT        = Chain of Thought，"一步步思考"
  tools      = 告诉模型有哪些工具可用
  tool_calls = 模型说"我要调用某个工具"
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

if "填" in API_KEY:
    raise SystemExit("请先填入 DeepSeek API key（第 30 行）")

from openai import OpenAI
client = OpenAI(api_key=API_KEY,base_url=BASE_URL)

# ============================================================
# Part 0 · Prompt 工程：三种问法的效果对比
# ============================================================
# 同一个任务（判断影评情感），三种 prompt 写法，看准确率差别。

print("=" * 60)
print("Part 0 · Prompt 工程三连")

reviews = [
    ("The plot was thin but the acting saved it.",        "mixed"),
    ("Absolutely loved it, watched it twice!",            "positive"),
    ("A complete waste of money and two hours.",          "negative"),
]

def ask(prompt, temperature=0.0):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user","content":prompt}],
        temperature=temperature,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()

# --- 方法 1：zero-shot（直接问，不给例子）---
print("\n--- 方法 1 · zero-shot ---")
zero_correct = 0
for text, truth in reviews:
    out = ask(f"判断这条影评的情感（positive/negative/mixed），只回答一个词：{text}")
    ok = truth in out.lower()
    zero_correct += ok
    print(f"  [{'✓' if ok else '✗'}] {out:12s} | 真实: {truth}")

# --- 方法 2：few-shot（给 3 个例子再问）---
print("\n--- 方法 2 · few-shot ---")
few_prompt = """判断影评情感，规则如下，只回答一个词：
例1: "Great fun from start to finish" → positive
例2: "Dull and lifeless, I gave up halfway" → negative
例3: "Some good jokes but a weak story" → mixed

现在判断: {text}"""
few_correct = 0
for text, truth in reviews:
    out = ask(few_prompt.format(text=text))
    ok = truth in out.lower()
    few_correct += ok
    print(f"  [{'✓' if ok else '✗'}] {out:12s} | 真实: {truth}")

# --- 方法 3：CoT（让它先推理再下结论）---
print("\n--- 方法 3 · CoT ---")
cot_out = ask(
    "判断这条影评的情感，先逐句分析在夸还是在骂，最后一行以'结论:'开头给出答案：\n"
    + reviews[0][0],
    temperature=0.0,
)
print(f"  {cot_out}")
print(f"  → 复杂任务上 CoT 明显更准，但更贵（输出 token 多）——这是权衡")

print(f"\n  zero-shot 对 {zero_correct}/3 | few-shot 对 {few_correct}/3")

# ============================================================
# Part 1 · Function Calling：先看模型"想调用"长什么样
# ============================================================
# 我们假装有一个天气查询工具。注意：是"假装"——函数就在本地，
# 模型根本看不到它的实现，只看到一份"说明书"（JSON Schema）。

print("=" * 60)
print("Part 1 · 看模型如何'点菜'")

# ---- 本地工具的真实实现（模型看不见这部分）----
MOCK_WEATHER = {"北京": (28, "晴"), "上海": (31, "多云"), "广州": (33, "雷阵雨")}

def get_weather(city: str) -> dict:
    """本地函数：查天气。这里用假数据，真实项目里换成 API 调用"""
    if city in MOCK_WEATHER:
        temp, cond = MOCK_WEATHER[city]
        return {"city": city, "temperature": temp, "condition": cond}
    return {"city": city, "error": "查不到该城市"}

# ---- 工具的"说明书"（模型只能看到这个）----
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市当前的天气和温度",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如'北京'"},
            },
            "required": ["city"],
        },
    },
}]

# ---- 第一次调用：模型返回的不是答案，而是"调用请求" ----
messages = [{"role": "user", "content": "北京今天多少度？"}]
resp = client.chat.completions.create(
    model=MODEL, messages=messages, tools=tools,
)

msg = resp.choices[0].message
print(f"  finish_reason: {resp.choices[0].finish_reason}")   # 应该是 "tool_calls"
print(f"  模型内容: {msg.content}")                          # None——它没直接回答
print(f"  它想调用: {msg.tool_calls[0].function.name}")
print(f"  参数 JSON: {msg.tool_calls[0].function.arguments}")

# ✅ 检查点：看清楚——模型只输出了一个 JSON 字符串，
#    什么都没执行。执行与否完全由你的代码决定。

# ============================================================
# Part 2 · 完整闭环：执行 → 回传 → 拿到最终回答
# ============================================================
# 四步循环：模型点菜 → 你做菜 → 菜端回去 → 模型总结

print("=" * 60)
print("Part 2 · 完整闭环")

# ① 把模型的"点菜"消息加回历史（必须！否则第二轮对话模型会懵）
messages.append(msg)

# ② 你的代码执行函数
tool_call = msg.tool_calls[0]
args = json.loads(tool_call.function.arguments)   # '{"city": "北京"}' → dict
result = get_weather(**args)
print(f"  本地执行 get_weather({args}) → {result}")

# ③ 把执行结果以 role="tool" 回传
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,       # 必须对应上，模型才知道这是哪次调用的结果
    "content": json.dumps(result, ensure_ascii=False),
})

# ④ 第二次调用：模型拿到结果，组织成人话
final = client.chat.completions.create(model=MODEL, messages=messages)
print(f"  模型最终回答: {final.choices[0].message.content}")


# ============================================================