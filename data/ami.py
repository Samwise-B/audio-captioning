import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from datasets import load_dataset
import whisper


class Ami(Dataset):
    def __init__(self, split: str, chunk_size: int = 30, max_speakers: int = 10):
        self.split = split
        self.ds = load_dataset("edinburghcstr/ami", "ihm", trust_remote_code=True)
        self.ds = self.ds[split]
        self.tk = WhisperTokenizer.from_pretrained("openai/whisper-base")
        self.extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
        self.max_speakers = max_speakers
        self.speaker_labels = [f"<|speaker_{i}|>" for i in range(1, max_speakers + 1)]
        self.tk.add_tokens(self.speaker_labels, special_tokens=True)
        self.max_chunk_size = 30 * 16000

    def __len__(self):
        return len(self.ds[self.split])

    def __getitem__(self, idx):
        """
        Combine multiple rows of audio and speakers until clip is 30 seconds long
        """
        row = self.ds[idx]
        row_count = 1
        meeting_id = row["meeting_id"]
        speaker_id = row["speaker_id"]
        speaker_count = 0
        speaker_to_id = {speaker_id: self.speaker_labels[speaker_count]}
        caption = f"{speaker_to_id[speaker_id]} {row["text"].lower()} "
        audio = torch.tensor(row["audio"]["array"])
        while len(audio) < self.max_chunk_size:
            print(meeting_id)
            print(speaker_to_id)
            print(speaker_count)
            print(audio.shape)
            print(caption)

            row = self.ds[idx + row_count]
            row_count += 1

            # check if new meeting
            if meeting_id != row["meeting_id"]:
                break

            # check if audio length exceeds max_size
            if len(audio) + len(row["audio"]["array"]) > self.max_chunk_size:
                break

            # add new speaker and check if
            speaker_id = row["speaker_id"]
            if not speaker_to_id.get(speaker_id):
                speaker_count += 1
                if speaker_count > self.max_speakers:
                    break

                speaker_to_id[speaker_id] = self.speaker_labels[speaker_count]

            new_audio = torch.tensor(row["audio"]["array"])
            audio = torch.cat((audio, new_audio), dim=0)
            caption += f"{speaker_to_id[speaker_id]} {row["text"].lower()} "

        audio_features = self.extractor(audio)
        return audio.unsqueeze(0), torch.tensor(self.tk.encode(caption)).unsqueeze(0)

    def collate_fn(batch):
        pass


if __name__ == "__main__":
    dataset = Ami("train")
    # dataloader = DataLoader(dataset, batch_size=1)
    model = whisper.load_model("tiny")
    row = dataset[0]
    pass
    # for batch in dataloader:
    #     continue
