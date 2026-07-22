"""Chat interface: upload a CSV (or use the bundled sample), ask questions in
plain English, get an answer plus an automatic chart.
"""

import pandas as pd
import streamlit as st

from agent import ask
from data_loader import load_data
from schema_context import build_schema_context

st.set_page_config(page_title="AI Data Analysis Agent", page_icon="\U0001F4CA")
st.title("AI Data Analysis Agent")
st.caption("Ask questions about a dataset in plain English. The agent writes and runs the SQL for you.")


@st.cache_data
def load_sample() -> pd.DataFrame:
    return load_data()


with st.sidebar:
    st.subheader("Dataset")
    uploaded = st.file_uploader("Upload a CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        table_name = "data"
        st.success(f"Loaded {len(df)} rows from {uploaded.name}")
    else:
        df = load_sample()
        table_name = "sales"
        st.info("Using the bundled Online Retail sample dataset.")

    st.dataframe(df.head(5), height=180)

schema = build_schema_context(df, table_name=table_name)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])
        if entry.get("chart_path"):
            st.image(entry["chart_path"])

question = st.chat_input("Ask a question about the data...")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, chart_path = ask(question, df, schema, table_name=table_name)
        st.write(answer)
        if chart_path:
            st.image(chart_path)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "chart_path": chart_path}
    )
