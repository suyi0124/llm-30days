from collections import Counter

def get_stats(vocab):
    """统计所有相邻符号对的出现次数。"""
    pairs = Counter()
    for word ,freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i],symbols[i+1])] += freq


    return pairs

def merge_vocab(pair, vocab):
    """把指定的一对符号合并成一个。"""
    out = {}
    bigram = " ".join(pair)             # ('e','s') → 'e s'
    replacement = "".join(pair)         # ('e','s') → 'es'
    for word, freq in vocab.items():
        out[word.replace(bigram, replacement)] = freq
    return out

vocab = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
}

print("=== Part 0：手写 BPE ===")
print("初始词表（空格分隔的是符号，</w> 表示词尾）：")

for w,f in vocab.items():
    print(f"{w!r:28s} 出现 {f} 次")
print()

for i in range(10):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"第 {i + 1} 次合并：{best[0]!r} + {best[1]!r} → {''.join(best)!r}  (出现 {pairs[best]} 次)")

print()
print("合并 10 次后的词表：")
for w,f in vocab.items():
    print(f"  {w!r:28s} 出现 {f} 次")
print()
print("看到了吗：'e'+'s' 先被合并（因为 est 出现 9 次最多），")
print("然后 'es'+'t' 合并成 'est'，再和 '</w>' 合并 —— 高频片段被自动提炼出来了。")
print("真实模型的词表就是这么从几十 GB 文本里跑出来的，只是合并 5 万次而不是 10 次。")
print()

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from transformers import AutoTokenizer

    print("=== Part 1：真实 tokenizer ===")

    # ① GPT-2 的英文 tokenizer（BPE，词表 50257）
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    en = "tokenization is fascinating, isn't it?"
    tokens = gpt2_tok.tokenize(en)
    print(f"英文原句: {en}")
    print(f"切成 {len(tokens)} 个 token: {tokens}")
    print(f"对应 id: {gpt2_tok.encode(en)}")
    print(f"还原: {gpt2_tok.decode(gpt2_tok.encode(en))}")
    print("注意 'tokenization' 被切成了 'token' + 'ization'，这就是子词（subword）")
    print()

except ImportError:
    print("=== Part 1 跳过：没装 transformers ===")
    print("想跑真实 tokenizer：pip install transformers")
    print("Part 0 的手写 BPE 不受影响，那才是今天的重点。")
except Exception as e:
    print(f"=== Part 1 出错（可能是网络下载失败）：{type(e).__name__}: {e}")
    print("可以换镜像重试，或先跳过——Part 0 已经覆盖了核心原理。")
