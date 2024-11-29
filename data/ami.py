import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

# import whisper


class Ami(Dataset):
    def __init__(
        self,
        split: str,
        subset_size: int = None,
        chunk_size: int = 30,
        max_speakers: int = 10,
    ):
        self.split = split
        self.ds = load_dataset("edinburghcstr/ami", "ihm", trust_remote_code=True)
        self.ds = self.ds[split]
        self.subset_size = subset_size

        self.tk = WhisperTokenizer.from_pretrained("openai/whisper-base")
        self.extractor = WhisperFeatureExtractor()

        self.max_speakers = max_speakers
        self.speaker_labels = [f"<|speaker_{i}|>" for i in range(1, max_speakers + 1)]
        self.tk.add_tokens(self.speaker_labels, special_tokens=True)
        self.tk_to_id = self.tk.convert_tokens_to_ids
        self.speaker_labels = [
            self.tk_to_id(f"<|speaker_{i}|>") for i in range(1, max_speakers + 1)
        ]
        self.max_chunk_size = chunk_size * 16000

    def __len__(self):
        if self.subset_size:
            return len(self.ds.select(range(self.subset_size)))
        else:
            return len(self.ds)

    def __getitem__(self, idx):
        """
        Combine multiple rows of audio and speakers until clip is 30 seconds long
        """
        # print("index:", idx)
        row = self.ds[idx]
        row_count = 1
        meeting_id = row["meeting_id"]
        speaker_id = row["speaker_id"]
        speaker_count = 0
        speaker_to_id = {speaker_id: self.speaker_labels[speaker_count]}

        caption = []

        if self.ds[idx - 1]["meeting_id"] != meeting_id:
            caption = (
                [self.tk_to_id("<|startoftranscript|>")]
                + [self.tk_to_id("<|en|>")]
                + [self.tk_to_id("<|transcribe|>")]
                + [self.tk_to_id("<|notimestamps|>")]
            )

        caption += [speaker_to_id[speaker_id]] + self.tk.encode(
            f" {row['text'].lower()} ", add_special_tokens=False
        )
        audio = torch.tensor(row["audio"]["array"])

        while len(audio) < self.max_chunk_size:
            # print(meeting_id)
            # print(speaker_to_id)
            # print(speaker_count)
            # print(audio.shape)
            # print(caption)

            row = self.ds[idx + row_count]
            row_count += 1

            # check if new meeting
            if meeting_id != row["meeting_id"]:
                caption += [self.tk_to_id("<|endoftext|>")]
                # add end of transcript
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
            caption += [speaker_to_id[speaker_id]] + self.tk.encode(
                f" {row['text'].lower()} ", add_special_tokens=False
            )

        # print(caption)
        caption = torch.tensor(caption)
        preprocessed = self.extractor(
            audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        audio_inpt = preprocessed["input_features"]
        audio_mask = preprocessed["attention_mask"]
        cap_inpt = caption[:-1]
        cap_targ = caption[1:]
        # audio_features = audio_features[]
        return audio_inpt, audio_mask, cap_inpt, cap_targ, cap_targ.shape[-1]

    def collate_fn(batch):
        audios, masks, cap_inpts, cap_targs, cap_lens = zip(*batch)

        audio_batch = torch.cat(audios, dim=0)
        mask_batch = torch.cat(masks, dim=0)
        stacked_targs = torch.cat(cap_targs, dim=0)
        padded_cap_inpts = pad_sequence(cap_inpts, batch_first=True)

        # cap_lens = [len(t) for t in cap_targs]

        return audio_batch, mask_batch, padded_cap_inpts, stacked_targs, cap_lens


if __name__ == "__main__":
    dataset = Ami("train")
    row = dataset[0]
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Ami.collate_fn)

    for batch in dataloader:
        break
