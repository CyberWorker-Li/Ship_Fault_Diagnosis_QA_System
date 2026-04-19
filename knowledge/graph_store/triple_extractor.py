import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests


RELATION_WHITELIST = {
    "发生于", "涉及部件", "导致", "表现为", "触发告警", "适用工况", "检测参数", "诊断步骤",
    "维修步骤", "需要工具", "更换备件", "风险提示", "前置条件", "后续验证", "来源案例", "证据片段",
}


@dataclass
class TripleExtractorConfig:
    api_key: str | None = None
    api_url: str = os.getenv("AI_ASSISTANT_LLM_API_URL", "http://localhost:11434/v1/chat/completions")
    model: str = os.getenv("AI_ASSISTANT_LLM_ANSWER_MODEL", "deepseek-r1:7b")
    timeout: int = int(os.getenv("AI_ASSISTANT_LLM_TIMEOUT", "60"))


class TripleExtractor:
    def __init__(self, config: TripleExtractorConfig | object | None = None):
        if isinstance(config, TripleExtractorConfig):
            self.config = config
        elif config is not None and hasattr(config, "llm_api_key"):
            self.config = TripleExtractorConfig(
                api_key=str(getattr(config, "llm_api_key", "") or ""),
                api_url=str(getattr(config, "llm_api_url", "") or "http://localhost:11434/v1/chat/completions"),
                model=str(getattr(config, "llm_answer_model", "deepseek-r1:7b") or "deepseek-r1:7b"),
                timeout=int(getattr(config, "llm_timeout", 60) or 60),
            )
        else:
            self.config = TripleExtractorConfig(api_key=os.getenv("AI_ASSISTANT_LLM_API_KEY", ""))

    def extract_records(self, text: str, use_llm: bool = True) -> List[Dict[str, object]]:
        if use_llm:
            if not self.config.api_key and "localhost" not in self.config.api_url:
                pass
            try:
                return self._extract_by_llm(text)
            except Exception as e:
                print(f"[警告] LLM三元组抽取失败: {e}，跳过此文本块。")
                return []
        triples = self._extract_by_rules(text)
        return [
            {
                "subject": s,
                "predicate": p,
                "object": o,
                "subject_type": "",
                "object_type": "",
                "predicate_type": p,
                "confidence": 0.7,
            }
            for s, p, o in triples
        ]

    def extract(self, text: str, use_llm: bool = True) -> List[Tuple[str, str, str]]:
        records = self.extract_records(text, use_llm=use_llm)
        tuples: List[Tuple[str, str, str]] = []
        for r in records:
            s = str(r.get("subject", "")).strip()
            p = str(r.get("predicate", "")).strip()
            o = str(r.get("object", "")).strip()
            if s and p and o:
                tuples.append((s, p, o))
        return self._deduplicate(tuples)

    def extract_with_focus(self, text: str, focus_entities: List[str], use_llm: bool = False) -> List[Tuple[str, str, str]]:
        focused = self._focus_text(text, focus_entities)
        records = self.extract_records(focused, use_llm=use_llm)
        tuples: List[Tuple[str, str, str]] = []
        for r in records:
            s = str(r.get("subject", "")).strip()
            p = str(r.get("predicate", "")).strip()
            o = str(r.get("object", "")).strip()
            if s and p and o:
                tuples.append((s, p, o))
        return self._deduplicate(tuples)

    def _extract_by_llm(self, text: str) -> List[Dict[str, object]]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        system_prompt = (
            "你是船舶故障知识抽取专家。请抽取结构化三元组并输出JSON对象。"
            "关系类型必须优先使用白名单："
            + "、".join(sorted(RELATION_WHITELIST))
            + "。若无法匹配则用‘证据片段’。"
            "输出格式: {\"triples\":[{\"subject\":\"\",\"predicate\":\"\",\"object\":\"\",\"subject_type\":\"\",\"object_type\":\"\",\"predicate_type\":\"\",\"confidence\":0.0}]}。"
            "如果没有有效信息，返回 {\"triples\":[]}。"
        )
        user_prompt = f"请从以下文本中抽取三元组：\n\n---\n{text}\n---"

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "stream": False,
        }
        try:
            resp = requests.post(self.config.api_url, headers=headers, json=payload, timeout=self.config.timeout)
        except Exception as e:
            raise RuntimeError(f"三元组抽取请求失败: {e}")
        if resp.status_code != 200:
            raise RuntimeError(f"三元组抽取HTTP失败: {resp.status_code} | {resp.text[:180]}")
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"三元组抽取响应解析失败: {e}")

        # 尝试从LLM的输出中解析JSON
        try:
            # 预处理：修复小模型常见的畸形结构
            fixed_content = content.replace(']], [[', '], [').replace('], [', '],###,[').replace('], [', '],') 
            # 先保护正确的逗号，再清理掉多余的括号
            
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    data = json.loads(match.group(0))
                    raw_triples = data.get("triples", [])
                    # ... 后面保持解析逻辑
                except:
                    pass # 失败则进入正则
            
            if match:
                try:
                    data = json.loads(match.group(0))
                    raw = data.get("triples", [])
                    out: List[Dict[str, object]] = []
                    for item in raw:
                        if isinstance(item, dict):
                            s = str(item.get("subject", "")).strip()
                            p = str(item.get("predicate", "")).strip()
                            o = str(item.get("object", "")).strip()
                            if not (s and p and o):
                                continue
                            c = float(item.get("confidence", 0.8) or 0.8)
                            out.append({
                                "subject": s,
                                "predicate": p,
                                "object": o,
                                "subject_type": str(item.get("subject_type", "") or ""),
                                "object_type": str(item.get("object_type", "") or ""),
                                "predicate_type": str(item.get("predicate_type", p) or p),
                                "confidence": max(0.0, min(1.0, c)),
                            })
                        elif isinstance(item, list) and len(item) >= 3:
                            s, p, o = str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip()
                            if s and p and o:
                                out.append({
                                    "subject": s,
                                    "predicate": p,
                                    "object": o,
                                    "subject_type": "",
                                    "object_type": "",
                                    "predicate_type": p,
                                    "confidence": 0.75,
                                })
                    return out
                except Exception:
                    pass

            found = re.findall(r'\[\s*["\']([^"\'\]]+)["\']\s*,\s*["\']([^"\'\]]+)["\']\s*,\s*["\']([^"\'\]]+)["\']\s*\]', content)
            if found:
                return [
                    {
                        "subject": t[0].strip(),
                        "predicate": t[1].strip(),
                        "object": t[2].strip(),
                        "subject_type": "",
                        "object_type": "",
                        "predicate_type": t[1].strip(),
                        "confidence": 0.7,
                    }
                    for t in found
                ]

            lines = re.findall(r'([^\s,]+)\s*[,，\t-]\s*([^\s,]+)\s*[,，\t-]\s*([^\s,]+)', content)
            if lines:
                return [
                    {
                        "subject": l[0].strip(),
                        "predicate": l[1].strip(),
                        "object": l[2].strip(),
                        "subject_type": "",
                        "object_type": "",
                        "predicate_type": l[1].strip(),
                        "confidence": 0.65,
                    }
                    for l in lines
                ]

            return []
        except Exception:
            return []

    def _focus_text(self, text: str, focus_entities: List[str]) -> str:
        entities = [e for e in focus_entities if e]
        if not entities:
            return text
        sentences = re.split(r"(?<=[。！？!?；;])\s*", text)
        selected = [s for s in sentences if any(e in s for e in entities)]
        if selected:
            return " ".join(selected)
        return text

    def _extract_by_rules(self, text: str) -> List[Tuple[str, str, str]]:
        triples: List[Tuple[str, str, str]] = []
        normalized = re.sub(r"\s+", " ", text)
        patterns = [
            (r"([\u4e00-\u9fa5A-Za-z0-9_\-]+)是([\u4e00-\u9fa5A-Za-z0-9_\-]+)", "是"),
            (r"([\u4e00-\u9fa5A-Za-z0-9_\-]+)属于([\u4e00-\u9fa5A-Za-z0-9_\-]+)", "属于"),
            (r"([\u4e00-\u9fa5A-Za-z0-9_\-]+)用于([\u4e00-\u9fa5A-Za-z0-9_\-]+)", "用于"),
            (r"([\u4e00-\u9fa5A-Za-z0-9_\-]+)包含([\u4e00-\u9fa5A-Za-z0-9_\-]+)", "包含"),
            (r"([\u4e00-\u9fa5A-Za-z0-9_\-]+)基于([\u4e00-\u9fa5A-Za-z0-9_\-]+)", "基于"),
        ]
        for pattern, rel in patterns:
            for m in re.finditer(pattern, normalized):
                s, o = m.group(1).strip(), m.group(2).strip()
                if s and o and s != o:
                    triples.append((s, rel, o))
        if not triples:
            return []
        return self._deduplicate(triples)

    def _deduplicate(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        seen = set()
        out = []
        for t in triples:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out