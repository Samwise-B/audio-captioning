import whisper

import torch
from torch.utils.data import DataLoader
from jiwer import wer

from data.homegrown import HomegrownDataset
from training.sam_r_train import validate_batch
from training.utils import add_speaker_tokens_to_whisper, collate_fn

processor, model = add_speaker_tokens_to_whisper()
model.load_state_dict(torch.load("./weights/whisper_diarization_weights.pth", weights_only=True))

dataset = HomegrownDataset(split='validate')
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

def main():
    for batch in dataloader:
        audio = batch['audio']
        validate_batch(model, audio)

if __name__ == "__main__":
    main()