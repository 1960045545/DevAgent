from core.agent import Agent
from .result import ChatResult

class ChatService:
    def __init__(self, agent: Agent):
        self.agent = agent

    def chat(self, user_message: str) -> ChatResult:
        try:
            response = self.agent.chat(user_message)
            return ChatResult(
                text=response.text,
                success=True,
                events=[],
            )
        except Exception as exc:
            return ChatResult(
                text="模型服务暂时不可用，请稍后再试。",
                success=False,
                events=[],
                error=str(exc),
            )