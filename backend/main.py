import torch
import whisper
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from init_minio import init_minio
from utils.preprocess import clean_prediction

# from utils.chunk_audio import split_audio
import tempfile

print("about to connect")
minio_client = init_minio()
buckets = minio_client.list_buckets()
for bucket in buckets:
    print(f"Bucket: {bucket.name}, Created: {bucket.creation_date}")


app = FastAPI()
model = whisper.load_model("tiny")
tk = whisper.tokenizer.get_tokenizer(multilingual=True)
max_cap = 200

@app.post("/generate-subtitle/")
async def generate_subtitle(audio_file: UploadFile = File(...), task: str = Form(...)):
    """
    Endpoint to accept an audio file and return a fixed subtitle.
    """

    if task not in ("Translate", "Transcribe"):
        raise HTTPException(
            status_code=400, detail="Must choose a valid task (transcribe/translate)"
        )
    # save audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await audio_file.read())
        temp_file_path = temp_file.name

    # audio = whisper.load_audio(temp_file_path, 16000)
    caption = model.transcribe(temp_file_path, task=task)
    return JSONResponse(content={"subtitle": caption["text"]})
