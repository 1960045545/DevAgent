from core import agent
from core.tool_space import ToolSpec
from dotenv import load_dotenv
from manager.model_provider_manager import ModelProviderConfig
import os

load_dotenv()


def add(a: int, b: int) -> int:
    return a + b


add_tool = ToolSpec(
    name="add",
    description="计算两个整数的和",
    parameters={
        "type": "object",
        "properties": {
            "a": {
                "type": "integer",
                "description": "第一个数字",
            },
            "b": {
                "type": "integer",
                "description": "第二个数字",
            },
        },
        "required": ["a", "b"],
    },
)


def build_model_providers() -> list[ModelProviderConfig]:
    providers: list[ModelProviderConfig] = []

    if (
        os.getenv("SILICON_BASE_URL")
        and os.getenv("SILICON_LLM_MODEL_ID")
    ):
        providers.append(
            ModelProviderConfig(
                name="silicon",
                base_url=os.getenv("SILICON_BASE_URL"),
                api_key=os.getenv("SILICON_API_KEY"),
                chat_model_id=os.getenv("SILICON_LLM_MODEL_ID"),
                history_model_id=os.getenv(
                    "SILICON_HISTORY_ABSTRACT_MODEL_ID"
                ),
                profile_model_id=os.getenv(
                    "SILICON_USER_PROFILE_MODEL_ID"
                ),
            )
        )

    if (
        os.getenv("BAI_LIAN_BASE_URL")
        and os.getenv("BAI_LIAN_LLM_MODEL_ID")
    ):
        providers.append(
            ModelProviderConfig(
                name="bailian",
                base_url=os.getenv("BAI_LIAN_BASE_URL"),
                api_key=os.getenv("BAI_LIAN_API_KEY"),
                chat_model_id=os.getenv("BAI_LIAN_LLM_MODEL_ID"),
                history_model_id=os.getenv(
                    "BAI_LIAN_HISTORY_ABSTRACT_MODEL_ID"
                ),
                profile_model_id=os.getenv(
                    "BAI_LIAN_USER_PROFILE_MODEL_ID"
                ),
            )
        )

    if (
        os.getenv("OLLAMA_BASE_URL")
        and os.getenv("OLLAMA_MODEL_ID")
    ):
        ollama_model_id = os.getenv("OLLAMA_MODEL_ID")
        providers.append(
            ModelProviderConfig(
                name="ollama",
                base_url=os.getenv("OLLAMA_BASE_URL"),
                api_key=os.getenv("OLLAMA_API_KEY"),
                chat_model_id=ollama_model_id,
                history_model_id=ollama_model_id,
                profile_model_id=ollama_model_id,
            )
        )

    return providers


if __name__ == "__main__":
    model_providers = build_model_providers()

    agent = agent.Agent(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        model_id=os.getenv("LLM_MODEL_ID"),
        providers=model_providers or None,
        max_retries=3,
        retry_interval=5,
        cooldown_seconds=30,
        max_tokens=3000,
        timeout=120,
    )

    agent.register_tool(add_tool, add)

    while True:
        user_msg = input("请输入内容（输入 end 结束）：")

        if user_msg == "end":
            print("结束输入")
            break

        res = agent.chat(user_msg)
        print(res.text)
