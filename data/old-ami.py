from datasets import load_dataset
import numpy as np
from torch.utils.data import IterableDataset, DataLoader

ds = load_dataset("edinburghcstr/ami", "ihm", trust_remote_code=True)


class AMIDataset(IterableDataset):
    def __init__(
        self,
        hf_ds,
        split="train",
        limit=None,
        chunk_length=480000,
        processor=None,
    ):
        self.chunk_length = chunk_length
        self.processor = processor
        full_ds = hf_ds[split]
        if limit != None:
            indices = range(min(limit, len(full_ds)))
            self.ds = full_ds.select(indices)
        else:
            self.ds = full_ds

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        current_item_index = 0
        samples = []
        full_text = ""

        while len(samples) < self.chunk_length:
            item = self.ds[current_item_index]
            audio = item["audio"]["array"]
            samples += audio
            speaker = item["speaker_id"]
            text = item["text"]
            full_text += text
            current_item_index += 1

        input_features = self.processor.feature_extractor(
            samples, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)

        labels = self.processor.tokenizer(
            f"<speaker_{speaker}> {text}", return_tensors="pt"
        ).input_ids.squeeze(0)

        yield {"input_features": input_features, "labels": labels}


if __name__ == "__main__":
    from transformers import WhisperProcessor
    import torch
    from torch.utils.data import DataLoader

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    ami_dataset = AMIDataset(ds, split="train", limit=5, processor=processor)

    dataloader = DataLoader(dataset=ami_dataset, batch_size=1, collate_fn="REMOVED")

    # Get first batch
    try:
        batch = next(iter(dataloader))

        print("Input features shape:", batch["input_features"].shape)
        print("Labels shape:", batch["labels"].shape)

        # To see the actual text:
        decoded_text = processor.decode(batch["labels"][0], skip_special_tokens=True)
        print("\nTranscript:", decoded_text)

        # If you want to see the spectrogram values:
        print(
            "\nSpectrogram min/max:",
            batch["input_features"][0].min().item(),
            batch["input_features"][0].max().item(),
        )
    except Exception as e:
        print(f"Error getting first batch: {str(e)}")
