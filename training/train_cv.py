from torch.utils.data import DataLoader
from datasets import load_dataset
from pathlib import Path
import sys

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from data.voice import CommonVoiceDataset

# Wrap the streamed dataset
dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "en",
    split="train[:10]",
    streaming=True,
    trust_remote_code=True,
)

# wrapped_dataset = CommonVoiceDataset(dataset)
# dataloader = DataLoader(wrapped_dataset, batch_size=1)

for batch in dataset["train"]:
    audio, transcript = batch
    print(audio.shape, transcript)
