import torch
import whisper
from utils import clean_prediction
from pathlib import Path
from chunk_audio import split_audio

max_cap = 200

model = whisper.load_model("tiny")
tk = whisper.tokenizer.get_tokenizer(multilingual=True)

hello_me_transcription = "Hello, my name is Izaak."
input_ids = []
input_ids += [tk.sot]
input_ids += [tk.language_token]
input_ids += [tk.transcribe]
input_ids += [tk.no_timestamps]
input_ids += tk.encode(hello_me_transcription)
input_ids += [tk.eot]

audio = whisper.load_audio("example_code/30s.m4a")
# audio = whisper.pad_or_trim(audio)
# mel = whisper.log_mel_spectrogram(audio)

chunks = split_audio(audio)
input_ids = [tk.sot, tk.language_token, tk.transcribe, tk.no_timestamps]
input_tks = torch.tensor(input_ids).unsqueeze(0).to(model.device)

for chunk in chunks:
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    input_mel = mel.unsqueeze(0).to(model.device)

    with torch.inference_mode():
        while input_tks.size(-1) <= max_cap:
            initial_predictions = model(tokens=input_tks, mel=input_mel)
            (
                top_k,
                top_k_indices,
            ) = torch.topk(initial_predictions[0, -1], 50, dim=-1)

            probs = torch.nn.functional.softmax(top_k, dim=-1)
            top_k_id = torch.multinomial(top_k, 1)
            next_token = top_k_indices[top_k_id]
            input_tks = torch.cat((input_tks, next_token.unsqueeze(0)), dim=-1)

            if next_token == tk.eot:
                break

caption = tk.decode(input_tks.squeeze().tolist())
caption = clean_prediction(caption)
