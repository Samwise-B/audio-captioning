import logging
from torch.utils.data import Dataset, DataLoader
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

        self.chunks = self._precompute_chunks()
        # Update self.chunk_count to be len(self.chunks)
        self.chunk_count = len(self.chunks)

    def _precompute_chunks(self):
        logging.info("Precomputing dataset chunks")
        print("Precomputing dataset chunks")
        chunks = []
        current_chunk = {'audio': [], 'text': [], 'meeting_id': None, 'prev_speaker': None}
        current_audio_length = 0
        current_meeting_id = None
        for row in self.ds:
            audio = row['audio']['array']
            text = row['text']
            meeting_id = row['meeting_id']
            speaker_id = row['speaker_id']

            if current_meeting_id is None or meeting_id != current_meeting_id or (current_audio_length + len(audio) > self.max_chunk_size):
                if current_chunk['audio']:
                    chunks.append(current_chunk)
                current_chunk = {'audio': [audio], 'text': [text], 'meeting_id': meeting_id, 'prev_speaker': speaker_id}
                current_audio_length = len(audio)
                current_meeting_id = meeting_id
            else:
                current_chunk['audio'].append(audio)
                if speaker_id != current_chunk['prev_speaker']:
                    current_chunk['text'].append(f" {self.speaker_label} {text}")
                else:
                    current_chunk['text'].append(text)
                current_audio_length += len(audio)
                current_chunk['prev_speaker'] = speaker_id

        if current_chunk['audio']:
            chunks.append(current_chunk)
        print(f"finished precomputing chunks. Length: {len(chunks)}")
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        aggregated_audio = np.concatenate(chunk['audio'])
        aggregated_text = ' '.join(chunk['text']).lower()
        meeting_id = chunk['meeting_id']
        return {
            'text': aggregated_text,
            'audio': aggregated_audio,
            'meeting_id': meeting_id,
        }

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
                "text": texts,
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