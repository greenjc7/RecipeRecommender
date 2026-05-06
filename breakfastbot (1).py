import streamlit as st
import os
import requests

st.set_page_config(
    page_title="Food.com Recipe Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main { background-color: #FAFAFA; }

    .header-banner {
        background: linear-gradient(135deg, #C62828 0%, #E53935 60%, #FF7043 100%);
        border-radius: 14px;
        padding: 26px 32px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 4px 20px rgba(198,40,40,0.25);
    }
    .header-banner h1 {
        margin: 0;
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        margin: 8px 0 0;
        opacity: 0.88;
        font-size: 0.95rem;
    }

    /* Hide sidebar toggle */
    [data-testid="collapsedControl"] { display: none; }

    /* Example prompt buttons */
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        background: white;
        color: #333;
        font-size: 0.85rem;
        padding: 10px 12px;
        transition: all 0.2s ease;
        text-align: left;
        white-space: normal;
        height: auto;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #E53935;
        color: #C62828;
        background: #FFF5F5;
        box-shadow: 0 2px 8px rgba(229,57,53,0.12);
        transform: translateY(-1px);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 4px 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9E9E9E;
        font-size: 0.78rem;
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px solid #EEEEEE;
    }
    .footer a { color: #E53935; text-decoration: none; }
    .footer a:hover { text-decoration: underline; }

    /* Section label */
    .section-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #757575;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load credentials from Streamlit secrets ───────────────────────────────────
databricks_host  = st.secrets.get("DATABRICKS_HOST",  os.getenv("DATABRICKS_HOST",  "https://dbc-0726d26f-3749.cloud.databricks.com"))
databricks_token = st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN", ""))
endpoint_name    = st.secrets.get("SERVING_ENDPOINT",  os.getenv("SERVING_ENDPOINT", "agents_isa632_7474656346303369-greenjc7-foodcom_recipe_recommen"))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Source:** Food.com (Kaggle)")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>🍳 Food.com Breakfast Recipe Recommender</h1>
  <p>Start your morning right! Ask me anything and I'll find the perfect breakfast recipe from Food.com.</p>
</div>
""", unsafe_allow_html=True)

# ── Endpoint caller ───────────────────────────────────────────────────────────
def call_serving_endpoint(host: str, token: str, endpoint: str, messages: list) -> str:
    """
    Calls the Mosaic Model Serving endpoint.
    Uses the Responses API format matching agent.py's predict() signature.
    Timeout bumped to 120s to survive cold cluster starts.
    """
    url = f"{host.rstrip('/')}/serving-endpoints/{endpoint}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"input": messages}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        # Responses API shape:
        # {"output": [{"content": [{"type": "output_text", "text": "..."}]}]}
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

    except requests.exceptions.Timeout:
        return (
            "⏱️ **Request timed out.** The Databricks cluster may be warming up — "
            "please try again in 15–30 seconds."
        )
    except requests.exceptions.HTTPError as e:
        return f"⚠️ HTTP {e.response.status_code}: {e.response.text[:300]}"
    except Exception as e:
        return f"⚠️ Error: {e}"

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Example prompts (only rendered when chat is empty) ───────────────────────
if not st.session_state.messages:
    st.markdown('<div class="section-label">💡 Try asking</div>', unsafe_allow_html=True)
    examples = [ "🥞 Easy pancake recipe for beginners",
        "🫐 Healthy overnight oats or granola ideas",
        "🥚 Quick breakfast under 15 minutes",
        "💪 High-protein breakfast meal prep",
        "🧇 What can I make with eggs and cheese?",
        "⭐ Top-rated breakfast casseroles on Food.com",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        with cols[i % 3]:
            if st.button(ex, use_container_width=True, key=f"ex_{i}"):
                st.session_state._prefill = ex
                st.rerun()

# ── FIX: pop _prefill OUTSIDE the messages-empty gate ────────────────────────
# Previously this lived inside `if not st.session_state.messages`, so any click
# after the first message was sent would silently discard the prefill value.
user_input = st.session_state.pop("_prefill", None)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🧑‍🍳" if msg["role"] == "user" else "🍽️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Chat input (overrides prefill if the user types something) ───────────────
chat_input = st.chat_input("Ask for a breakfast recipe or anything morning food-related…")
if chat_input:
    user_input = chat_input

# ── Send message ──────────────────────────────────────────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍🍳"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🍽️"):
        with st.spinner("🔍 Finding recipes…"):
            if not databricks_host or not databricks_token:
                answer = (
                    "⚠️ **Configuration error.** Credentials are missing — "
                    "please check your `.streamlit/secrets.toml` file."
                )
            else:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-6:]
                ]
                answer = call_serving_endpoint(
                    databricks_host, databricks_token, endpoint_name, history
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div class='footer'>"
    "🍽️ Food.com Recipe Recommender &nbsp;|&nbsp; ISA 632 RAG Project &nbsp;|&nbsp; Miami University &nbsp;|&nbsp;"
    "Carson Green, Taylor Martin, Gabe Bjork, Billy Knapp &nbsp;|&nbsp;"
    "<a href='https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions' target='_blank'>Kaggle Dataset</a>"
    "</div>",
    unsafe_allow_html=True,
)
