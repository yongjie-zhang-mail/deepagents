"""DeepSeek deepseek-v4-flash model demo via langchain-deepseek."""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

load_dotenv()


def main() -> None:
    model = ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )

    print("=== basic invoke ===")
    response = model.invoke([HumanMessage(content="Hello")])
    print(response.content)

    print("\n=== streaming (count to 3) ===")
    for chunk in model.stream([HumanMessage(content="Count to 3.")]):
        print(chunk.content, end="", flush=True)
    print()

    print("\n=== math reasoning ===")
    response = model.invoke([HumanMessage(content="What is 7 * 8? Reply with the number only.")])
    print(response.content)


if __name__ == "__main__":
    main()
