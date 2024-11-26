import torch
import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import clean_prediction
import tempfile

app = FastAPI()
model = whisper.load_model("tiny")
tk = whisper.tokenizer.get_tokenizer(multilingual=True)
max_cap = 200


@app.post("/generate-subtitle/")
async def generate_subtitle(audio_file: UploadFile = File(...)):
    """
    Endpoint to accept an audio file and return a fixed subtitle.
    """
    # You can add logic to process the audio file here.
    # For now, we're returning a fixed subtitle.

    # save audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await audio_file.read())
        temp_file_path = temp_file.name

    audio = whisper.load_audio(temp_file_path, 16000)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    input_ids = [tk.sot, tk.language_token, tk.transcribe, tk.no_timestamps]

    input_tks = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    input_mel = mel.unsqueeze(0).to(model.device)

    with torch.inference_mode():
        while input_tks.size(-1) <= max_cap:
            initial_predictions = model(tokens=input_tks, mel=input_mel)
            probs = torch.nn.functional.softmax(initial_predictions[0, -1])
            next_token = torch.multinomial(probs, 1)
            input_tks = torch.cat((input_tks, next_token.unsqueeze(0)), dim=-1)

            if next_token == tk.eot:
                break

    caption = tk.decode(input_tks.squeeze().tolist())
    caption = clean_prediction(caption)
    return JSONResponse(content={"subtitle": caption})
