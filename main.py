from core import agent
from dotenv import load_dotenv
import os
load_dotenv()

if __name__ == "__main__":
    agent = agent.Agent(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        model_id=os.getenv("MODEL_ID"),
        max_tokens=3000,
        timeout=120
    )
    while True:
        user_msg = input("请输入内容（输入 end 结束）：")
        if user_msg == "end":
            print("结束输入")
            break
        for chunk in agent.stream_chat(user_msg):
            print(chunk, end="", flush=True)
        print()
