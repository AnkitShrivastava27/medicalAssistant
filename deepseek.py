import os
import streamlit as st
import re
from huggingface_hub import InferenceClient


api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_key:
    st.error("API key not found. Please add it in Streamlit → Settings → Secrets.")
    st.stop()


client = InferenceClient(api_key=api_key)

st.set_page_config(page_title="Medical Assistant AI", page_icon="💬")
st.title("💬 Medical Assistant - AI Chatbot")

user_input = st.text_input("Ask a medical question...", key="input")

if user_input:
  
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional medical assistant. Only respond to medical-related questions. "
                "If the question is not medical, simply respond: 'I'm only able to assist with medical-related questions.'"
            )
        },
        {"role": "user", "content": user_input}
    ]

  
    try:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="HuggingFaceH4/zephyr-7b-beta",  # ✅ Free to use model
                messages=messages,
                stream=False
            )
            final_answer = response.choices[0].message.content
            clean_answer = re.sub(r"<think>.*?</think>", "", final_answer, flags=re.DOTALL).strip()
            st.markdown(f"**Assistant:** {clean_answer}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
