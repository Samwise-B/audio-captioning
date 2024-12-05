import os
import wandb

import torch
from torch.utils.data import DataLoader

from data.ami import Ami
from models.CustomWhisper import CustomWhisper
from training.validation import validate_batch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project='whisper-diarization', job_type='inference')
    custom_model_wrapper = CustomWhisper(base_model="openai/whisper-tiny", max_speakers=5, numbered_speakers=False)
    model = custom_model_wrapper.model

    # Use wandb artifact
    artifact = wandb.use_artifact('whisper-diarization/whisper_model:latest', type='model')
    artifact_dir = artifact.download()

    # Load model weights from artifact
    weights_file = os.path.join(artifact_dir, 'whisper_diarization_ami.pth')
    state_dict = torch.load(weights_file, map_location=torch.device(device))
    model.load_state_dict(state_dict)

    dataset = Ami(split='validation', subset_size=100)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Ami.get_collate_fn(dataset.tk, dataset.extractor))
    for batch in dataloader:
        audio = batch['input_features']
        validate_batch(model, audio, batch['text'], dataset.tk, numbered_speakers=False)

if __name__ == "__main__":
    main()