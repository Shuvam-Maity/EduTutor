import streamlit as st
import requests
import time

# Page config
st.set_page_config(
    page_title="EduTutor",
    page_icon="📚",
    layout="centered"
)

# Header
st.title("📚 EduTutor")
st.markdown("**Curriculum Q&A powered by fine-tuned Mistral-7B**")
st.markdown("---")

# Sidebar info
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    EduTutor is a fine-tuned LLM for curriculum Q&A.
    
    **Stack:**
    - Mistral-7B fine-tuned with QLoRA
    - AWQ 4-bit quantization (71% smaller)
    - Served via TGI on HuggingFace
    - FastAPI inference layer
    - Weave observability
    
    **Model:** [Shuvam-Maity/edututor-mistral-awq](https://huggingface.co/Shuvam-Maity/edututor-mistral-awq)
    """)

# API endpoint
API_URL = "https://shuvam-maity-edututor-api.hf.space/generate"

# Input
question = st.text_area(
    "Ask a curriculum question:",
    placeholder="e.g. What is the process by which plants make food using sunlight?",
    height=100
)

# Submit button
if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                start    = time.time()
                response = requests.post(
                    API_URL,
                    json={"question": question, "max_tokens": 200},
                    timeout=60
                )
                data    = response.json()
                latency = round((time.time() - start) * 1000)

                st.markdown("### Answer")
                st.success(data["answer"])

                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Latency", f"{latency}ms")
                with col2:
                    st.metric("Model", "Mistral-7B AWQ")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("The TGI server may be sleeping. Try again in a moment.")

# Footer
st.markdown("---")
st.markdown(
    "Built by [Shuvam Maity](https://linkedin.com/in/shuvam-maity) · "
    "[GitHub](https://github.com/Shuvam-Maity/EduTutor) · "
    "[HuggingFace](https://huggingface.co/Shuvam-Maity)"
)