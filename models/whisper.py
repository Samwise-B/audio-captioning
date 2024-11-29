import torch
import torch.nn as nn
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)


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


if __name__ == "__main__":
    model = Whisper()
    pass
