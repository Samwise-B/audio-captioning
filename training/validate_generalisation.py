import whisper

import torch
from torch.utils.data import DataLoader

from data.homegrown import HomegrownDataset
from models.CustomWhisper import CustomWhisper
from training.sam_r_train import validate_batch
from training.utils import collate_fn

def main():
    custom_model_wrapper = CustomWhisper(base_model="openai/whisper-tiny", max_speakers=5)
    model = custom_model_wrapper.model
    model.load_state_dict(torch.load("./weights/whisper_diarization_v3.pth", weights_only=True))
    dataset = HomegrownDataset(split='validate')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        audio = batch['audio']
        validate_batch(model, audio, batch['texts'], custom_model_wrapper.tokenizer)

if __name__ == "__main__":
    main()