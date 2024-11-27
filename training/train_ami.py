import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import sys

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from data.ami import Ami
from models.whisper import Whisper

ds = Ami("train")
dataloader = DataLoader(ds, batch_size=5, collate_fn=Ami.collate_fn)

model = Whisper()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
for i, (audio, masks, cap_inpt, cap_targ, cap_lens) in tqdm(
    enumerate(dataloader), desc="Training"
):
    optimizer.zero_grad()
    out = model(audio, masks, cap_inpt)
    pred = out.logits

    cropped_pred = torch.cat([x[: cap_lens[i]] for i, x in enumerate(pred)], dim=0)
    loss = criterion(cropped_pred, cap_targ)
    loss.backward()
    print(f"loss: {loss.item()}")
