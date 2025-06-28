import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import streamlit as st
import re
# Load API key
load_dotenv()
client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

# Streamlit UI
st.title("💬 Medical Assistant -AI")
user_input = st.text_input("Ask a medical question...", key="input")

if user_input:
    # Construct restricted medical-only messages
    messages = [
        {
            "role": "system",
            "content": (
            "You are a professional medical assistant. Only respond to medical-related questions. "
            "Do not include any internal thoughts or explanations about your reasoning process. "
            "Provide short, clear replies. If the question is not medical, simply respond: "
            "'I'm only able to assist with medical-related questions.'"
        )

        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    # Non-streaming call — wait for full completion
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-0528",
        messages=messages,
        stream=False,
    )
    final_answer = response.choices[0].message.content

# Remove <think>...</think> block using regex
    clean_answer = re.sub(r"<think>.*?</think>", "", final_answer, flags=re.DOTALL).strip()

# Display it
    st.markdown(f"**Assistant:** {clean_answer}")
   

