import streamlit as st
import requests

# URL of the FastAPI backend
API_URL = "http://backend:8000/generate-subtitle/"

st.title("Audio Captioning & Translation")
st.write("Upload an audio file and get its subtitles or translate it into english.")

# File uploader widget
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
left, right = st.columns([3, 1], vertical_alignment="center")
with left:
    option = st.selectbox(
        "",
        ("Transcribe", "Translate"),
    )
caption = None

if uploaded_file is not None:
    # Display uploaded file details
    st.audio(uploaded_file, format="audio/wav")

right.markdown(
    """
    <h6>
    </h6>
    """,
    unsafe_allow_html=True,
)

if right.button("Upload"):
    # Send the file to the backend
    with st.spinner("Generating subtitle..."):
        response = requests.post(
            API_URL,
            data={"task": option},
            files={"audio_file": uploaded_file.getvalue()},
        )

    if response.status_code == 200:
        # Display the subtitle
        caption = response.json().get("subtitle", "No subtitle returned.")

    else:
        # Handle errors
        st.error("Failed to generate subtitle. Please try again.")

title = st.text_area("Output", caption)
