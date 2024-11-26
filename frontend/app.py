import streamlit as st
import requests

# URL of the FastAPI backend
API_URL = "http://backend:8000/generate-subtitle/"

st.title("Audio to Subtitle Generator")
st.write("Upload an audio file and get its subtitles.")

# File uploader widget
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
subtitle = None

if uploaded_file is not None:
    # Display uploaded file details
    st.audio(uploaded_file, format="audio/wav")

    # Send the file to the backend
    with st.spinner("Generating subtitle..."):
        response = requests.post(
            API_URL,
            files={"audio_file": uploaded_file.getvalue()},
        )

    if response.status_code == 200:
        # Display the subtitle
        subtitle = response.json().get("subtitle", "No subtitle returned.")

    else:
        # Handle errors
        st.error("Failed to generate subtitle. Please try again.")

st.write(f"**Subtitle:** {subtitle}")
