"""The core text-to-SQL agent: turns a natural language question into a
safe SQL query, executes it, and returns the result.
"""

import ollama
import pandas as pd

from data_loader import load_data
from query_engine import format_result, run_safe_query
from schema_context import build_schema_context

MODEL = "qwen2.5:7b"
MAX_TOOL_ROUNDS = 3

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Run a read-only SQL SELECT query against the 'sales' table and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT statement."},
                },
                "required": ["sql"],
            },
        },
    }
]


def _system_prompt(schema: str) -> str:
    return (
        "You are a data analyst assistant. Answer questions about the dataset "
        "by calling the run_sql_query tool - never guess numbers yourself. "
        "Only the columns listed below exist; never invent column names.\n\n"
        f"{schema}"
    )


def ask(question: str, df: pd.DataFrame, schema: str) -> str:
    messages = [
        {"role": "system", "content": _system_prompt(schema)},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)

        if not response.message.tool_calls:
            return response.message.content

        messages.append(response.message)
        for call in response.message.tool_calls:
            sql = call.function.arguments["sql"]
            try:
                result = run_safe_query(df, sql)
                content = format_result(result)
            except Exception as e:
                # Fed back to the model as a tool result, not raised - this lets
                # the model see its own mistake and try a corrected query next.
                content = f"Query failed: {e}"
            messages.append({"role": "tool", "content": content, "tool_name": call.function.name})

    return "Could not produce an answer within the tool-call limit."


if __name__ == "__main__":
    data = load_data()
    schema = build_schema_context(data)

    questions = [
        "What is the total revenue in this dataset?",
        "Which country has the highest number of orders?",
        "What was the average unit price across all transactions?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask(q, data, schema)}\n")
