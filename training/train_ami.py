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
from utils.inference import run_inference

torch.manual_seed(42)

ds = Ami("train", subset_size=1)
dataloader = DataLoader(ds, batch_size=1, collate_fn=Ami.collate_fn)

model = Whisper()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

running_loss = []
running_accuracy = []
for epoch in range(1000):
    for i, (audio, masks, cap_inpt, cap_targ, cap_lens) in enumerate(dataloader):
        optimizer.zero_grad()
        out = model(audio, masks, cap_inpt)
        pred = out.logits

        cropped_pred = torch.cat([x[: cap_lens[i]] for i, x in enumerate(pred)], dim=0)
        loss = criterion(cropped_pred, cap_targ)
        loss.backward()
        optimizer.step()

        precision = sum(torch.argmax(cropped_pred, dim=-1) == cap_targ) / len(cap_targ)
        print(f"epoch: {epoch} loss: {loss.item()}, precision: {precision}", end="\r")

    if (epoch + 1) % 10 == 0:
        o, t = run_inference(model, ds, 0, 100)
        print(f"Inference: {o}")
        print("=======================")
        print(f"Target: {t}")
