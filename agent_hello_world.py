"""Minimal proof that tool-calling works before building the real data agent."""

import ollama

MODEL = "qwen2.5:7b"


def add_numbers(a: float, b: float) -> float:
    return a + b


# This is the "menu" of tools the model is allowed to use. The model never
# runs code itself - it only ever replies with "please call add_numbers(a=128, b=347)"
# and our own Python code does the actual calculation below.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "first number"},
                    "b": {"type": "number", "description": "second number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]

AVAILABLE_FUNCTIONS = {"add_numbers": add_numbers}


def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]

    response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)

    if not response.message.tool_calls:
        return response.message.content

    # The model asked for a tool call instead of answering directly.
    # We execute the requested function ourselves and hand the result back.
    messages.append(response.message)
    for call in response.message.tool_calls:
        function = AVAILABLE_FUNCTIONS[call.function.name]
        result = function(**call.function.arguments)
        messages.append(
            {"role": "tool", "content": str(result), "tool_name": call.function.name}
        )

    final_response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
    return final_response.message.content


if __name__ == "__main__":
    question = "What is 128 plus 347?"
    print(f"Question: {question}")
    print(f"Answer:   {run_agent(question)}")
