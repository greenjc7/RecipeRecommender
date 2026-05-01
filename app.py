"""
Food.com Recipe Recommender — Streamlit App
Calls the deployed Mosaic Model Serving endpoint.
"""

import streamlit as st
import os
import requests

st.set_page_config(
    page_title="Food.com Recipe Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    .header-banner {
        background: linear-gradient(135deg, #C62828 0%, #E53935 60%, #FF7043 100%);
        border-radius: 14px; padding: 22px 30px; margin-bottom: 20px; color: white;
    }
    .header-banner h1 { margin: 0; font-size: 1.9rem; }
    .header-banner p  { margin: 6px 0 0; opacity: 0.88; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    databricks_host = st.text_input(
        "Databricks Host",
        value=os.getenv("DATABRICKS_HOST", ""),
        placeholder="https://<workspace>.azuredatabricks.net",
    )
    databricks_token = st.text_input(
        "Databricks Token",
        value=os.getenv("DATABRICKS_TOKEN", ""),
        placeholder="dapi...",
        type="password",
    )
    # Use Cell 14 output from DB notebook
    endpoint_name = st.text_input(
        "Model Serving Endpoint Name",
        value=os.getenv("SERVING_ENDPOINT", "agents_isa632_7474656346303369-greenjc7-foodcom_recipe_recommen"),
        help="Printed in Cell 14 of the Databricks notebook as 'Endpoint URL'.",
    )
    st.markdown("---")
    st.markdown("**Source:** Food.com (Kaggle)  \n**Recipes:** 5,000 of 231,637 avaliable  \n**Reviews:** 5,000 of 1,132,367 avaliable")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>🍽️ Food.com Recipe Recommender</h1>
  <p>Ask me anything — I'll find the perfect recipe from 231,000+ Food.com dishes.</p>
</div>
""", unsafe_allow_html=True)

# ── Endpoint caller ───────────────────────────────────────────────────────────
def call_serving_endpoint(host: str, token: str, endpoint: str, messages: list) -> str:
    """
    Calls the Mosaic Model Serving endpoint deployed in Lab 4 / Cell 13.
    Uses the Responses API format matching agent.py's predict() signature.
    """
    url = f"{host.rstrip('/')}/serving-endpoints/{endpoint}/invocations"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"input": messages}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        # Responses API shape: {"output": [{"content": [{"type": "output_text", "text": "..."}]}]}
        if "output" in data:
            for item in data["output"]:
                content = item.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if c.get("type") == "output_text":
                            return c.get("text", "").strip()
                elif isinstance(content, str):
                    return content.strip()

        # Fallback: standard chat completions shape
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()

        return str(data)

    except requests.exceptions.HTTPError as e:
        return f"⚠️ HTTP {e.response.status_code}: {e.response.text[:300]}"
    except Exception as e:
        return f"⚠️ Error: {e}"

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Example prompts ───────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    examples = [
        "🥗 Quick vegetarian dinner under 30 minutes",
        "🎂 Easy chocolate cake for a beginner",
        "🍗 High-protein meal prep for the week",
        "🌮 Mexican recipes with chicken and beans",
        "🥦 What can I make with broccoli and garlic?",
        "⭐ Top-rated pasta dishes on Food.com",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            if st.button(ex, use_container_width=True, key=f"ex_{i}"):
                st.session_state._prefill = ex
                st.rerun()
    user_input = st.session_state.pop("_prefill", None)
else:
    user_input = None

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🧑‍🍳" if msg["role"] == "user" else "🍽️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
chat_input = st.chat_input("Ask for a recipe, substitution, or anything food-related…")
if chat_input:
    user_input = chat_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍🍳"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🍽️"):
        with st.spinner("🔍 Finding recipes…"):
            if not databricks_host or not databricks_token:
                answer = (
                    "⚠️ **Configuration needed.** Enter your Databricks Host and Token in the sidebar. "
                    "The Model Serving Endpoint Name should match what was printed in Cell 14 of the Databricks notebook."
                )
            else:
                # Pass last 6 turns as conversation history
                history = [{"role": m["role"], "content": m["content"]}
                           for m in st.session_state.messages[-6:]]
                answer = call_serving_endpoint(databricks_host, databricks_token, endpoint_name, history)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9E9E9E;font-size:0.8rem;'>"
    "🍽️ Food.com Recipe Recommender &nbsp;|&nbsp; ISA 632 Group Project &nbsp;||&nbsp; Carson Green, Taylor Martin, Billy Knapp, Gabe Bjork;|&nbsp; Miami University &nbsp;|&nbsp;"
    "<a href='https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions' target='_blank'>Kaggle Dataset</a>"
    "</p>",
    unsafe_allow_html=True,
)
