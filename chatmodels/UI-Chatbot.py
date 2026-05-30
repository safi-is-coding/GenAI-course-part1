from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Mood ChatBot",
    page_icon="🤖"
)

# ---------------- MODEL ---------------- #

model = init_chat_model("groq:llama-3.1-8b-instant")

# ---------------- AI MODES ---------------- #

ai_modes = {
    "Funny 😂": "You are a funny AI agent who always replies humorously.",
    
    "Sad 😢": "You are a sad AI agent who replies emotionally and sadly.",
    
    "Angry 😡": "You are an angry AI agent who replies aggressively."
}

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("⚙️ AI Settings")

    selected_mode = st.selectbox(
        "Choose AI Mode",
        list(ai_modes.keys())
    )

    st.markdown("---")

    st.write(f"Current Mode: {selected_mode}")

# ---------------- SESSION STATE ---------------- #

if "messages" not in st.session_state:

    st.session_state.messages = [
        SystemMessage(
            content=ai_modes[selected_mode]
        )
    ]

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# ---------------- UPDATE MODE ---------------- #

if (
    st.session_state.messages[0].content
    != ai_modes[selected_mode]
):

    st.session_state.messages = [
        SystemMessage(
            content=ai_modes[selected_mode]
        )
    ]

    st.session_state.chat_history = []

# ---------------- TITLE ---------------- #

st.title("🤖 AI Mood ChatBot")

st.write("Type `0` to end the conversation.")

# ---------------- DISPLAY CHAT ---------------- #

for sender, message in st.session_state.chat_history:

    if sender == "user":

        st.chat_message("user").write(message)

    else:

        st.chat_message("assistant").write(message)

# ---------------- USER INPUT ---------------- #

prompt = st.chat_input("Type your message...")

# ---------------- CHATBOT RESPONSE ---------------- #

if prompt:

    # Exit Conversation
    if prompt == "0":

        st.warning("Conversation Ended 👋")
        st.stop()

    # Store User Message
    st.session_state.chat_history.append(
        ("user", prompt)
    )

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # AI Response
    with st.spinner("AI is typing..."):

        response = model.invoke(
            st.session_state.messages
        )

    # Store AI Response
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

    st.session_state.chat_history.append(
        ("assistant", response.content)
    )

    st.rerun()