import json
import math
import re
from collections import Counter


def tokenize(text: str):
    """中文按 单字+二元组 切分，英文数字按词切分"""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    chars = re.findall(r'[\u4e00-\u9fff]', text)
    tokens += chars
    tokens += [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return tokens


class KnowledgeBase:
    """纯Python实现的BM25知识库检索，替代Elasticsearch"""

    def __init__(self, data_file: str, k1: float = 1.5, b: float = 0.75):
        with open(data_file, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # 顶层兼容：dict 取 values，list 直接用
        if isinstance(raw, dict):
            items = list(raw.values())
        else:
            items = raw

        self.entries = []
        skipped = 0
        for item in items:
            if not isinstance(item, dict):
                skipped += 1
                continue
            q = item.get('question') or item.get('instruction') or ''
            a = item.get('answer') or item.get('output') or ''
            q, a = q.strip(), a.strip()
            if q and a:
                self.entries.append({'question': q, 'answer': a})
            else:
                skipped += 1

        if self.entries:
            print(f"样例 -> 问: {self.entries[0]['question'][:30]} | 答: {self.entries[0]['answer'][:30]}")
        print(f"跳过 {skipped} 条（字段缺失或答案为空）")
        print(f"知识库加载完成，共 {len(self.entries)} 条")

        # ===== 构建 BM25 索引（这部分之前被误删了） =====
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(e['question']) for e in self.entries]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.doc_freqs = [Counter(t) for t in self.doc_tokens]

        df = Counter()
        for tokens in self.doc_tokens:
            for t in set(tokens):
                df[t] += 1
        n = len(self.doc_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 3):
        """返回 [(score, entry), ...]，按分数降序"""
        if not self.entries:
            return []
        q_tokens = tokenize(query)
        scores = []
        for i, tf in enumerate(self.doc_freqs):
            score = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                f = tf[t]
                idf = self.idf.get(t, 0)
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl)
                score += idf * f * (self.k1 + 1) / denom
            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(scores[i], self.entries[i]) for i, _ in ranked[:top_k]]
