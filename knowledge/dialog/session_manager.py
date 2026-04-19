from collections import deque
from typing import Deque, Dict, List


class SessionManager:
    def __init__(self, max_turns: int = 6):
        self.history: Deque[Dict[str, str]] = deque(maxlen=max_turns)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.history.append({"user": user_text, "assistant": assistant_text})

    def recent_turns(self) -> List[Dict[str, str]]:
        return list(self.history)

    def context_text(self) -> str:
        blocks = []
        for item in self.history:
            blocks.append(f"用户: {item['user']}\n助手: {item['assistant']}")
        return "\n".join(blocks)