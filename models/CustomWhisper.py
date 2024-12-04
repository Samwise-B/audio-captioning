import torch
import torch.nn as nn
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)


class CustomWhisper(nn.Module):
    def __init__(self, max_speakers: int = 10, base_model="openai/whisper-tiny", numbered_speakers=True):
        super().__init__()

        self.processor = WhisperProcessor.from_pretrained(base_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(
           base_model
        )

        # print("Model attributes:", dir(self.model))
        # print("\nModel named parameters:")
        # for name, param in self.model.named_parameters():
        #     print(name)
    
        if(numbered_speakers):
            self.new_tokens = {
                "additional_special_tokens": [
                    f"<|speaker_{i+1}|>" for i in range(max_speakers)
                ]
            }
        else:
            self.new_tokens = {
                "additional_special_tokens": [
                    "<|speaker_change|>"
                ]
            }
        self.tokenizer = WhisperTokenizer.from_pretrained(base_model)
        self.tokenizer.add_special_tokens(self.new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))


        self.special_token_indices = [
            self.tokenizer.convert_tokens_to_ids(token)
            for token in self.new_tokens["additional_special_tokens"]
        ]

         # Freeze all parameters
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Only enable grads for decoder embeddings
        # embed_tokens = self.model.model.decoder.embed_tokens
        # for param in embed_tokens.parameters():
        #     param.requires_grad = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device:{device}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, audio, mask, captions):
        pred = self.model(audio, mask, captions)
        return pred


if __name__ == "__main__":
    model = CustomWhisper()
    pass
