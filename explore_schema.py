"""Demonstrates why the schema context from schema_context.py matters.

Asks the model the same question twice: once with no information about the
data, once with the schema. The first answer is a guess with made-up column
names; the second uses the real ones. This is the difference between a model
that can write a working query and one that can't.
"""

import ollama

from data_loader import load_data
from schema_context import build_schema_context

MODEL = "qwen2.5:7b"
QUESTION = "Which exact columns would you use to calculate total revenue in this dataset, and how?"


def ask_without_context() -> str:
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": QUESTION}])
    return response.message.content


def ask_with_context(schema: str) -> str:
    system_prompt = (
        "You are a data analyst assistant. You only know about the dataset "
        "through the schema below - never invent column names that aren't listed.\n\n"
        f"{schema}"
    )
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QUESTION},
        ],
    )
    return response.message.content


if __name__ == "__main__":
    schema = build_schema_context(load_data())

    print("=== Without schema context ===")
    print(ask_without_context())

    print("\n=== With schema context ===")
    print(ask_with_context(schema))
