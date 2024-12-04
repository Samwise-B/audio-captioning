from torch.utils.data import Dataset, DataLoader, Subset
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class Ami(Dataset):
    def __init__(
        self,
        split: str,
        subset_size: int = None,
        chunk_size: int = 30,  # Max duration in seconds for a chunk
    ):
        self.split = split
        self.ds = load_dataset("edinburghcstr/ami", "ihm", trust_remote_code=True, split=f"{split}[:{subset_size}]" if subset_size else split)
        
        # self.ds = self.ds[split]
        self.subset_size = subset_size

        self.tk = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        self.extractor = WhisperFeatureExtractor()

        self.speaker_label = "<|speaker_change|>"
        self.tk.add_tokens([self.speaker_label], special_tokens=True)
        self.tk_to_id = self.tk.convert_tokens_to_ids
        self.max_chunk_size = chunk_size * 16000  # Sampling rate is 16000 Hz

        self.pointer = 0  # Maintain a pointer to track dataset position
        self.chunk_count = self._precompute_chunks()
        self.prev_speaker = ""

    def _precompute_chunks(self):
        """
        Precompute the number of chunks in the dataset based on the grouping logic.
        """
        total_chunks = 0
        current_meeting_id = None
        current_audio_length = 0

        for row in self.ds:
            audio = row["audio"]["array"]
            meeting_id = row["meeting_id"]

            if (
                meeting_id != current_meeting_id
                or current_audio_length + len(audio) > self.max_chunk_size
            ):
                # Start a new chunk
                total_chunks += 1
                current_meeting_id = meeting_id
                current_audio_length = len(audio)
            else:
                # Add to the current chunk
                current_audio_length += len(audio)

        # Apply subset size if specified
        if self.subset_size:
            return min(total_chunks, self.subset_size)
        return total_chunks

    def __len__(self):
        return self.chunk_count

    def __getitem__(self, idx):
        """
        Aggregate rows until:
        - The meeting ID changes
        - The total audio length exceeds max_chunk_size
        """
        # Ignore idx for sequential access
        aggregated_audio = np.array([])
        aggregated_text = ""
        if self.pointer >= len(self.ds):  # Check if pointer exceeds dataset length
            raise IndexError("Dataset pointer exceeds length")

        current_meeting_id = self.ds[self.pointer]["meeting_id"]

        while self.pointer < len(self.ds):
            row = self.ds[self.pointer]
            audio = row["audio"]["array"]
            text = row["text"]
            meeting_id = row["meeting_id"]

            # Stop if meeting ID changes or audio length exceeds max_chunk_size
            if meeting_id != current_meeting_id or len(aggregated_audio) + len(audio) > self.max_chunk_size:
                break

            # Append current row's data
            aggregated_audio = np.concatenate((aggregated_audio, audio))

            if self.pointer > 0 and row["speaker_id"] != self.prev_speaker:
                aggregated_text += f" <|change_speaker|> {text}"
            elif self.pointer > 0:
                aggregated_text += f" { text}"
            else:
                aggregated_text += text

            self.pointer += 1
            self.prev_speaker = row["speaker_id"]


        return {
            "text": aggregated_text.lower(),
            "audio": aggregated_audio,
            "meeting_id": current_meeting_id,
        }

    def reset_pointer(self):
        """
        Reset the pointer to the beginning of the dataset.
        """
        self.pointer = 0

    @staticmethod
    def get_collate_fn(tokenizer, extractor  = WhisperFeatureExtractor()):
        def collate_fn(batch):
            audio = [item["audio"] for item in batch]
            texts = [item["text"] for item in batch]
            input_features = extractor(audio, sampling_rate=16000, return_tensors="pt").input_features

            tokenized_texts = tokenizer(
                texts, 
                padding=True, 
                truncation=True,
                return_tensors="pt"
            )

            return {
                "input_ids": tokenized_texts.input_ids,
                "attention_mask": tokenized_texts.attention_mask,
                "input_features": input_features,
            }
        return collate_fn


if __name__ == "__main__":
    dataset = Ami("train", subset_size=1000)
    print(f"Number of chunks: {len(dataset)}")
    tokenizer = dataset.tk
    extractor = dataset.extractor

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Ami.get_collate_fn(tokenizer, extractor))

    for batch in dataloader:
        print(f"Batch Input IDs: {batch['input_ids'].shape}")
        print(f"Batch Audio Features: {batch['input_features'].shape}")
        print(f"Masks: {batch['attention_mask'].shape}")
        break