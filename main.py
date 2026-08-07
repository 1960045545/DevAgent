from core import agent
from core.tool_space import ToolSpec
from dotenv import load_dotenv
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


if __name__ == "__main__":
    agent = agent.Agent(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        model_id=os.getenv("MODEL_ID"),
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