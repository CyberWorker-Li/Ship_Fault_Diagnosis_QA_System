import re
from typing import List

from knowledge.shared.schemas import QueryPlan


STOPWORDS = {"什么", "什么是", "定义", "关系", "区别", "为什么", "如何", "请问", "一下", "吗", "？", "?", "有什么", "是什么"}


class NLUAnalyzer:
    def analyze(self, question: str) -> QueryPlan:
        entities = self._extract_entities(question)
        intent, mode, qtype = self._classify(question)
        return QueryPlan(
            intent=intent,
            entities=entities,
            question_type=qtype,
            retrieval_mode=mode,
            top_k=5,
        )

    def _extract_entities(self, question: str) -> List[str]:
        q = question.replace("？", "").replace("?", "").strip()
        if "什么是" in q:
            core = q.split("什么是", 1)[1].strip()
            if core:
                return [core[:20]]
        if "的定义" in q:
            core = q.split("的定义", 1)[0].strip()
            if core:
                return [core[:20]]

        relation_q = any(k in q for k in ["关系", "区别", "联系"])
        if relation_q:
            stripped = q
            for k in ["有什么关系", "是什么关系", "关系是什么", "之间", "有什么", "关系", "区别", "联系"]:
                stripped = stripped.replace(k, " ")
            for sep in ["和", "与", "及", "、", "跟"]:
                stripped = stripped.replace(sep, " ")
            parts = [x.strip() for x in re.split(r"\s+", stripped) if x.strip()]
            entities = [p[:20] for p in parts if len(p) >= 2]
            if entities:
                return entities[:4]

        if "为什么" in q:
            stripped = q.replace("为什么", " ")
            for k in ["适合", "用于", "能够", "可以", "在", "中", "里", "上", "?", "？"]:
                stripped = stripped.replace(k, " ")
            parts = [x.strip() for x in re.split(r"\s+", stripped) if x.strip()]
            entities = [p[:20] for p in parts if len(p) >= 2 and p not in STOPWORDS]
            if entities:
                return entities[:4]

        tokens = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9_]{2,}", q)
        uniq: List[str] = []
        for t in tokens:
            if t in STOPWORDS:
                continue
            if t.endswith("是什么"):
                t = t[:-3]
            if t not in uniq and t:
                uniq.append(t)
        return uniq[:6]

    def _classify(self, question: str) -> tuple[str, str, str]:
        q = question.strip()
        if any(k in q for k in ["关系", "区别", "联系", "路径"]):
            return "ask_relation", "graph_first", "multi_hop"
        if any(k in q for k in ["什么是", "定义", "概念"]):
            return "ask_concept", "vector_first", "simple"
        if any(k in q for k in ["为什么", "原理"]):
            return "ask_reason", "hybrid", "complex"
        return "ask_general", "hybrid", "simple"