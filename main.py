import streamlit as st
import base64
from io import BytesIO
from PIL import Image

from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://35.91.145.246:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


# wide mode
st.set_page_config(layout="centered")

st.title("Chat with Multi-modal LLMs - MMG")
st.subheader("Using Streamlit + vLLM")

# choose model
model = st.selectbox("Choose a model", ["llava-1.5-7b-hf"])
st.session_state["model"] = model

# chatbot stuff
st.markdown("---")


def process_image(uploaded_file):
    # You can perform additional processing on the image here if needed
    # For example, using PIL to resize the image
    image = Image.open(uploaded_file)
    # resized_image = image.resize((300, 300))

    # Convert the processed image to a base64-encoded string
    buffered = BytesIO()
    image.save(
        buffered, format="PNG"
    )  # Save as PNG for simplicity, you can choose another format
    processed_image_b64 = "data:image/png;base64," + base64.b64encode(
        buffered.getvalue()
    ).decode("utf-8")
    # processed_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return processed_image_b64


# # init session state of the uploaded image
# image_b64 = upload_image()
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process the uploaded image if needed
    image_b64 = process_image(uploaded_file)

# image_b64 = convert_to_base64(uploaded_file)
# ask question
q = st.text_input("Ask a question about the image(s)")
if q:
    question = q
else:
    question = "Describe the image:"


if st.button("Generate Response"):
    with st.spinner("Generating..."):
        res = client.chat.completions.create(
            model=f"llava-hf/{model}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_b64}},
                    ],
                }
            ],
            max_tokens=300,
        )

        st.chat_message("question").markdown(f"**{question}**", unsafe_allow_html=True)
        st.chat_message("response").write(res.choices[0].message.content)

