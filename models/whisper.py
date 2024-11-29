import torch
import torch.nn as nn
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Whisper(nn.Module):
    def __init__(self, max_speakers: int = 10):
        super().__init__()

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base"
        )
        self.new_tokens = {
            "additional_special_tokens": [
                f"<|speaker_{i+1}|>" for i in range(max_speakers)
            ]
        }
        self.tokeniser = WhisperTokenizer.from_pretrained("openai/whisper-base")
        self.tokeniser.add_special_tokens(self.new_tokens)
        self.whisper.resize_token_embeddings(len(self.tokeniser))

    def forward(self, audio, mask, captions):
        pred = self.whisper(audio, mask, captions)
        return pred

    def run_inference(self, audio, max_len: int = 200):
        tk_to_id = self.tokeniser.convert_tokens_to_ids

        preprocessed = self.processor(
            audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )

        audio_inpt = preprocessed["input_features"]
        audio_mask = preprocessed["attention_mask"]

        with torch.inference_mode():
            # idx = random.randint(0, ds.__len__())
            audio, mask = audio_inpt.to(device), audio_mask.to(device)
            inpt = torch.tensor(
                [tk_to_id("<|startoftranscript|>")]
                + [tk_to_id("<|en|>")]
                + [tk_to_id("<|transcribe|>")]
                + [tk_to_id("<|notimestamps|>")],
                device=device,
            ).unsqueeze(0)
            for _ in range(max_len):
                out = model(audio, mask, inpt)
                pred = out.logits
                new_logit = pred[:, -1, :]
                next_token = torch.argmax(new_logit, dim=-1)
                inpt = torch.cat([inpt, next_token.unsqueeze(0)], dim=-1)

            return self.tokeniser.decode(inpt.squeeze(), skip_special_tokens=False)


if __name__ == "__main__":
    model = Whisper()
    pass
