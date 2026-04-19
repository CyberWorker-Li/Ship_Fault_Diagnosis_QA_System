import json
import re
from pathlib import Path
from typing import List

from knowledge.shared.config import Settings
from knowledge.shared.schemas import Evidence


def _tail_chars(text: str, max_chars: int) -> str:
    text = str(text or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


REQUIRED_KEYS = [
    "故障现象", "风险警告", "故障原因", "故障排查流程", "维修步骤", "验收标准", "维护与预防建议", "结论"
]


class RagAnswerer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def answer(self, question: str, evidences: List[Evidence], intent: str = "ask_general", conversation_context: str = "") -> str:
        if not evidences:
            return "根据当前知识库，我暂时找不到相关证据。"

        try:
            ctx = str(conversation_context or "").strip()
            if ctx and hasattr(self.settings, "session_max_chars"):
                ctx = _tail_chars(ctx, int(getattr(self.settings, "session_max_chars", 0) or 0)).strip()
            generated = self._llm_answer(question, evidences, intent, conversation_context=ctx)
            if self.settings.enable_llm_polish and self.settings.polish_base_url and self.settings.polish_api_key:
                try:
                    generated = self._polish_text(generated)
                except Exception:
                    pass
            if self.settings.concise_answer:
                return generated

            top = evidences[:3]
            lines = [f"问题：{question}", "参考证据："]
            for i, e in enumerate(top, 1):
                snippet = e.content[:140].replace("\n", " ")
                source = Path(e.source_ref or "unknown").name
                lines.append(f"[{i}] {snippet}（来源: {source}）")
            lines.append("综合回答：" + generated)
            return "\n".join(lines)
        except Exception as e:
            return f"综合回答生成失败：{e}"

    def _extract_json(self, text: str) -> dict:
        txt = str(text or "").strip()
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    def _to_list(self, v) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    def _valid_structured(self, data: dict) -> tuple[bool, list[str]]:
        missing = []
        for k in REQUIRED_KEYS:
            vals = self._to_list(data.get(k, []))
            if not vals:
                missing.append(k)
        return len(missing) == 0, missing

    def _render_structured(self, data: dict) -> str:
        lines = []
        for k in REQUIRED_KEYS:
            lines.append(f"### {k}")
            vals = self._to_list(data.get(k, []))
            if vals:
                for v in vals:
                    lines.append(f"- {v}")
            else:
                lines.append("- 证据不足，暂无法给出该项结论。")
            lines.append("")
        return "\n".join(lines).strip()

    def _llm_answer(self, question: str, evidences: List[Evidence], intent: str, conversation_context: str = "") -> str:
        if not self.settings.enable_llm_answer:
            raise RuntimeError("LLM回答开关未启用")
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI
        except Exception as e:
            raise RuntimeError(f"缺少LangChain依赖，请安装 langchain-openai/langchain-core: {e}")

        evidence_lines = []
        for i, e in enumerate(evidences, 1):
            source = Path(e.source_ref or "unknown").name
            evidence_lines.append(f"[{i}] {e.content}\n（来源: {source}）")

        history_block = ""
        if conversation_context:
            history_block = (
                "\n对话历史（仅用于理解指代与省略，不作为事实来源；如与证据冲突，以证据为准）：\n"
                + conversation_context
                + "\n"
            )

        prompt = (
            "请严格基于证据输出JSON，禁止输出JSON外文本。"
            "必须包含以下键："
            "故障现象, 风险警告, 故障原因, 故障排查流程, 维修步骤, 验收标准, 维护与预防建议, 结论。"
            "每个键的值必须是数组，数组每条都要附带引用编号如[1][2]。"
            "若证据不足，也要保留该键，并写‘证据不足’。\n"
            f"意图: {intent}\n问题: {question}"
            + history_block
            + "\n证据:\n"
            + "\n".join(evidence_lines)
            + "\n输出示例: {\"故障现象\":[\"... [1]\"],\"风险警告\":[\"... [2]\"],\"故障原因\":[\"... [1][3]\"],\"故障排查流程\":[\"... [1]\"],\"维修步骤\":[\"... [1]\"],\"验收标准\":[\"... [1]\"],\"维护与预防建议\":[\"... [2]\"],\"结论\":[\"... [1][2]\"]}"
        )

        llm = ChatOpenAI(
            model=self.settings.llm_answer_model,
            api_key=self.settings.llm_api_key or "no-key",
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout,
            temperature=0.1,
        )
        system = "你是船舶故障诊断助手。必须可追溯、强结构化输出。"

        first = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        content = str(first.content).strip()
        data = self._extract_json(content)
        ok, missing = self._valid_structured(data)

        if not ok:
            repair_prompt = (
                "你的上一次输出缺少这些必填键：" + "、".join(missing) + "。"
                "请仅返回完整JSON，所有键必须存在且值为数组。"
                "缺失项请填‘证据不足[1]’。"
            )
            second = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt + "\n\n" + repair_prompt)])
            data = self._extract_json(str(second.content).strip())

        return self._render_structured(data)

    def _polish_text(self, text: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=self.settings.polish_model,
            api_key=self.settings.polish_api_key or "no-key",
            base_url=self.settings.polish_base_url,
            timeout=self.settings.llm_timeout,
            temperature=0.2,
        )
        result = llm.invoke([
            SystemMessage(content="你是中文写作润色助手，在不改变事实并保留所有引用编号如[1][2]的前提下优化语言表达。只输出润色后的文本。"),
            HumanMessage(content=text),
        ])
        return str(result.content).strip() or text