import os
import librosa
import torch
import whisper
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from minio_client import MinioClientWrapper
from models.whisper import CustomWhisper
from utils.preprocess import clean_prediction
from transformers import WhisperForConditionalGeneration

import tempfile

print("about to connect")
minio = MinioClientWrapper()
buckets = minio.client.list_buckets()
for bucket in buckets:
    print(f"Bucket: {bucket.name}, Created: {bucket.creation_date}")

diarization_model = CustomWhisper(base_model="openai/whisper-tiny", max_speakers=5)
weights_data = minio.load_weights("whisper_diarization", "v3")
if weights_data:
    state_dict = torch.load(weights_data)

    diarization_model.model.load_state_dict(state_dict)

    # Print total number of parameters
    total_params = sum(p.numel() for p in diarization_model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in diarization_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print model structure with parameter shapes
    for name, param in diarization_model.named_parameters():
        print(f"{name}: {param.shape}")
        
    # Print size in MB
    param_size = sum(p.numel() * p.element_size() for p in diarization_model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in diarization_model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    print(f"Model size: {size_mb:.2f} MB")
    print("Successfully loaded custom weights into model")


model = whisper.load_model("tiny")
tk = whisper.tokenizer.get_tokenizer(multilingual=True)

app = FastAPI()
max_cap = 200

def get_start_input_ids(tokenizer):
    input_ids = []

    sot_token = '<|startoftranscript|>'
    sot_token_id = tokenizer.convert_tokens_to_ids(sot_token)

    input_ids += [sot_token_id]

    language_token = '<|en|>'
    language_token_id = tokenizer.convert_tokens_to_ids(language_token)

    input_ids += [language_token_id]

    no_timestamps_token = '<|notimestamps|>'
    no_timestamps_token_id = tokenizer.convert_tokens_to_ids(no_timestamps_token)
    input_ids += [no_timestamps_token_id]
    return input_ids

eot_token = '<|endoftranscript|>'

def validate(model, input_mel, tokenizer):
    eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)
    model.eval()
    with torch.no_grad():
        #  TODO replace with something better
        input_ids = get_start_input_ids(tokenizer)
        input_tkns_tensor = torch.tensor(input_ids).unsqueeze(0).to(model.device)

        for i in range(80):
            initial_predictions = model(decoder_input_ids=input_tkns_tensor, input_features=input_mel)
            # 
            next_tkn = torch.argmax(initial_predictions.logits, dim=-1)[0,-1].unsqueeze(0)
            input_tkns_tensor = torch.cat((input_tkns_tensor.squeeze(), next_tkn), dim=0).unsqueeze(0)
            if input_tkns_tensor[-1, -1].item() == eot_token_id:
                break

    decoded_initial_output = tokenizer.decode(input_tkns_tensor.squeeze().tolist())
    return decoded_initial_output

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

@app.post("/diarize/")
async def process_audio(audio_file: UploadFile = File(...), task: str = Form(...)):
    """
    Endpoint to accept an audio file and return a diarized transcript
    """
    if task not in ("Translate", "Transcribe"):
        raise HTTPException(
            status_code=400, detail="Must choose a valid task (transcribe/translate)"
        )

    # save audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await audio_file.read())
        temp_file_path = temp_file.name

    try:
        # Load the audio file using librosa
        audio, sampling_rate = librosa.load(temp_file_path, sr=16000)
        processor = diarization_model.processor
        # Process the audio
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
        print(f"inputs: {inputs}")
        
        # # Generate transcription
        # generated_ids = diarization_model.model.generate(inputs["input_features"])
        # print(f"generated ids{generated_ids}")
        
        # # Decode the generated ids
        # transcription = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        

        transcription = validate(diarization_model.model, inputs['input_features'], diarization_model.tokenizer)
        print(f"transcription {transcription}")
        
        return JSONResponse(content={"subtitle": transcription})
        
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)
