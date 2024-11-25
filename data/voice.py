from torch.utils.data import IterableDataset
import torch


class CommonVoiceDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        """
        This method will iterate through the streaming dataset and yield data examples.
        """
        for example in self.dataset:
            audio = example["audio"][
                "array"
            ]  # Assuming audio is stored as a numpy array
            transcript = example["sentence"]  # Text transcription for the audio
            yield torch.tensor(audio), transcript  # Yield a data example

    def __getitem__(self, idx):
        """
        Optional: If you want to access individual items by index, you can define __getitem__.
        """
        example = self.dataset[idx]  # Get one example at a time
        audio = example["audio"]["array"]
        transcript = example["sentence"]
        return torch.tensor(audio), transcript
