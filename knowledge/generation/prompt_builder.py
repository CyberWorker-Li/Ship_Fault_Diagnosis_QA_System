from pathlib import Path
from typing import List

from knowledge.shared.schemas import Evidence


def build_prompt(question: str, evidences: List[Evidence]) -> str:
    lines = [
        "你是船舶故障诊断智能助手。请严格基于证据作答，结论后标注引用编号。",
        f"问题: {question}",
        "证据:",
    ]
    for i, e in enumerate(evidences, 1):
        source = Path(e.source_ref).name if e.source_ref else "unknown"
        lines.append(f"[{i}] {e.content}（来源: {source}）")
    lines.append("输出要求: 必须包含固定章节（故障现象/风险警告/故障原因/故障排查流程/维修步骤/验收标准/维护与预防建议/结论），且每条都带引用编号如[1][2]。")
    return "\n".join(lines)