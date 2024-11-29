import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from utils.preprocess import clean_prediction
from models.whisper import Whisper
from minio import Minio
import os
from pathlib import Path

# from utils.chunk_audio import split_audio
import tempfile
import torchaudio

app = FastAPI()
max_cap = 200


@app.on_event("startup")
async def startup_event():
    minio_client = Minio(
        "minio:9000",
        access_key="admin",
        secret_key="development",
        secure=False,
    )
    global model_base
    global model_d
    global tk
    model_base = Whisper()
    model_d = Whisper()
    model_dir = Path(__file__).parent / "weights"
    """Download files from MinIO only if they do not exist locally."""
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        objects = minio_client.list_objects("weights")
        os.makedirs("weights", exist_ok=True)
        for obj in objects:
            file_path = os.path.join("weights", os.path.basename(obj.object_name))
            try:
                response = minio_client.get_object("weights", obj.object_name)
                with open(file_path, "wb") as file_data:
                    for data in response.stream(32 * 1024):
                        file_data.write(data)
                print(f"Downloaded {obj.object_name} to {file_path}")
            except Exception as e:
                raise Exception(f"Error downloading {obj.object_name}: {str(e)}")

    model_d.load_state_dict(
        torch.load(
            model_dir / "whisper-diarization-best-train.pt",
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    model_base.eval()
    model_d.eval()


@app.post("/generate-subtitle/")
async def generate_subtitle(audio_file: UploadFile = File(...), task: str = Form(...)):
    """
    Endpoint to accept an audio file and return a fixed subtitle.
    """

    if task not in ("Translate", "Transcribe", "Diarization"):
        raise HTTPException(
            status_code=400, detail="Must choose a valid task (transcribe/translate)"
        )
    # Save the uploaded audio file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await audio_file.read())
        temp_file_path = temp_file.name

    try:
        # Convert audio to the format expected by Whisper
        audio, rate = torchaudio.load(temp_file_path)  # Load the audio as a waveform
        if rate != 16000:  # Whisper expects a 16kHz sample rate
            audio = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)(
                audio
            )

        # Pass the processed audio to the model
        if task.lower() == "transcribe":
            result = model_base.transcribe(audio)
        elif task.lower() == "translate":
            result = model_base.transcribe(audio, task="translate")
        else:
            result = {"text": model_d.run_inference(audio)}

        # Return the transcription/translation
        return JSONResponse(content={"subtitle": result["text"]})

    finally:
        # Clean up temporary file
        os.remove(temp_file_path)
