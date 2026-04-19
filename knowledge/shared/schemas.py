from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class QueryPlan:
    intent: str
    entities: List[str]
    question_type: str
    retrieval_mode: str
    top_k: int = 5


@dataclass
class Evidence:
    evidence_id: str
    evidence_type: str
    content: str
    score: float
    source_ref: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)