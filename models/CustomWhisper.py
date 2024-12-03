import torch
import torch.nn as nn
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)


class CustomWhisper(nn.Module):
    def __init__(self, max_speakers: int = 10, base_model="openai/whisper-tiny"):
        super().__init__()

        self.processor = WhisperProcessor.from_pretrained(base_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(
           base_model
        )
        self.new_tokens = {
            "additional_special_tokens": [
                f"<|speaker_{i+1}|>" for i in range(max_speakers)
            ]
        }
        self.tokenizer = WhisperTokenizer.from_pretrained(base_model)
        self.tokenizer.add_special_tokens(self.new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, audio, mask, captions):
        pred = self.model(audio, mask, captions)
        return pred


if __name__ == "__main__":
    model = CustomWhisper()
    pass
