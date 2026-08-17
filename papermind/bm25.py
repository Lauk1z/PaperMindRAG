"""BM25 稀疏检索模块：纯 NumPy/标准库实现，零新增依赖。

BM25 是经典词频-逆文档频率打分：
    score(q, d) = Σ IDF(qi) * tf·(k1+1) / (tf + k1·(1-b+b·|d|/avgdl))
专业术语（PatchCore/MVTec/coreset）词面精确匹配是稠密向量的弱项，
BM25 恰好补上；两者用 RRF 融合（见 retriever.py）。

中文支持：不引分词库，CJK 连续段切字符 bigram（"异常检测"->异常/常检/检测），
词面召回足够且完全可解释。
"""
import math
import re
from collections import Counter
from typing import List, Tuple

_WORD = re.compile(r"[a-zA-Z0-9]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """英文小写化 + CJK 字符 bigram。"""
    tokens = [w.lower() for w in _WORD.findall(text)]
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    """针对小规模语料（万级块）的内存 BM25 索引。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._doc_tf: List[Counter] = []
        self._doc_len: List[int] = []
        self._df: Counter = {}          # 词 -> 出现过的文档数
        self._n_docs = 0
        self._avgdl = 0.0

    def build(self, texts: List[str]) -> "BM25Index":
        self._doc_tf, self._doc_len, self._df = [], [], Counter()
        for t in texts:
            toks = tokenize(t)
            self._doc_tf.append(Counter(toks))
            self._doc_len.append(len(toks))
        self._n_docs = len(texts)
        self._avgdl = (sum(self._doc_len) / self._n_docs) if self._n_docs else 0.0
        for tf in self._doc_tf:
            self._df.update(tf.keys())   # 每文档每词计一次 df
        return self

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """返回 (块索引, 分数) 降序；无词面重叠的文档不返回（分数为0）。"""
        if self._n_docs == 0:
            return []
        q_tokens = tokenize(query)
        scores: List[float] = [0.0] * self._n_docs
        for tok in q_tokens:
            df = self._df.get(tok)
            if not df:                    # 语料中未出现 -> 无信息量
                continue
            idf = math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))
            for i, tf_counter in enumerate(self._doc_tf):
                tf = tf_counter.get(tok)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b *
                                        self._doc_len[i] / (self._avgdl or 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(((i, s) for i, s in enumerate(scores) if s > 0),
                        key=lambda x: -x[1])
        return ranked[:top_k]
