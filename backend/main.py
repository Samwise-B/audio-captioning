import os
import librosa
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import torch
from minio_client import MinioClientWrapper
from models.utils import clean_prediction, infer
from models.CustomWhisper import CustomWhisper

import tempfile

print("about to connect")
minio = MinioClientWrapper()
buckets = minio.client.list_buckets()
for bucket in buckets:
    print(f"Bucket: {bucket.name}, Created: {bucket.creation_date}")

diarization_model = CustomWhisper(base_model="openai/whisper-tiny", max_speakers=5, numbered_speakers=False)
weights_data = minio.load_weights("whisper_diarization", "ami")
weights = torch.load(weights_data, map_location="cpu")
diarization_model.model.load_state_dict(weights)

app = FastAPI()
max_cap = 200

#  TODO use transformers package instead for this
# model = whisper.load_model("tiny")
# tk = whisper.tokenizer.get_tokenizer(multilingual=True)
# @app.post("/generate-subtitle/")
# async def generate_subtitle(audio_file: UploadFile = File(...), task: str = Form(...)):
#     """
#     Endpoint to accept an audio file and return a fixed subtitle.
#     """

#     if task not in ("Translate", "Transcribe"):
#         raise HTTPException(
#             status_code=400, detail="Must choose a valid task (transcribe/translate)"
#         )
#     # save audio file temporarily
#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         temp_file.write(await audio_file.read())
#         temp_file_path = temp_file.name

#     # audio = whisper.load_audio(temp_file_path, 16000)
#     caption = model.transcribe(temp_file_path, task=task)
#     return JSONResponse(content={"subtitle": caption["text"]})

@app.post("/diarize/")
async def process_audio(audio_file: UploadFile = File(...), task: str = Form(...)):
    """
    Endpoint to accept an audio file and return a diarized transcript
    """
    if task not in ("Translate", "Transcribe"):
        raise HTTPException(
            status_code=400, detail="Must choose a valid task (transcribe/translate)"
        )

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await audio_file.read())
        temp_file_path = temp_file.name

    try:
        audio, _ = librosa.load(temp_file_path, sr=16000)
        processor = diarization_model.processor

        inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
        print(f"inputs: {inputs}")
        
        # TODO: see if you can use this
        # # Generate transcription
        # generated_ids = diarization_model.model.generate(inputs["input_features"])
        # print(f"generated ids{generated_ids}")
        
        # # Decode the generated ids
        # transcription = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        transcription = infer(diarization_model.model, inputs['input_features'], diarization_model.tokenizer)
        transcription = clean_prediction(transcription)
        transcription_w_line_breaks = transcription.replace(" <|speaker_change|> ", "\n\n")
        print(f"transcription {transcription_w_line_breaks}")
        
        return JSONResponse(content={"subtitle": transcription_w_line_breaks})
        
    finally:
        os.unlink(temp_file_path)
