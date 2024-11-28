import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import sys
import wandb
import logging

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from data.ami import Ami
from models.whisper import Whisper
from utils.inference import run_inference

torch.manual_seed(42)
logging.basicConfig(
    filename="training.log",  # File to log to
    filemode="w",  # Overwrite file each time; use "a" to append
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

subset_size = 10
batch_size = 10
ds = Ami("train", subset_size=subset_size)
dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=Ami.collate_fn)

val_ds = Ami("validation", subset_size=subset_size)
val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=Ami.collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Whisper()
model.to(device)
model_name = "whisper-diarization"

print(
    f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

val_freq = (ds.__len__() // batch_size) / 1
num_epochs = 1000

config = {
    "batch_size": batch_size,
    "subset_size": subset_size,
    "learning_rate": 1e-4,
    "num_epochs": num_epochs,
    "val_freq": val_freq,
}

wandb.init(project="image-captioning", name=model_name, config=config)
running_loss = []
running_accuracy = []
for epoch in range(num_epochs):
    for i, (audio, masks, cap_inpt, cap_targ, cap_lens) in enumerate(dataloader):
        audio, masks, cap_inpt, cap_targ = (
            audio.to(device),
            masks.to(device),
            cap_inpt.to(device),
            cap_targ.to(device),
        )

        optimizer.zero_grad()
        out = model(audio, masks, cap_inpt)
        pred = out.logits

        cropped_pred = torch.cat(
            [x[: cap_lens[i]] for i, x in enumerate(pred)], dim=0, device=device
        )
        loss = criterion(cropped_pred, cap_targ)
        loss.backward()
        optimizer.step()

        precision = sum(torch.argmax(cropped_pred, dim=-1) == cap_targ) / len(cap_targ)
        running_loss.append(loss.item())
        running_accuracy.append(precision)
        # print(f"epoch: {epoch} loss: {loss.item()}, precision: {precision}", end="\r")

    if (epoch + 1) % val_freq == 0:
        logging.info("Running inference...")
        o, t = run_inference(model, ds, 0, 100)
        logging.info(f"Inference: \n{o}")
        logging.info("=======================")
        logging.info(f"Target: \n{t}")

        # validation loop
        with torch.inference_mode():
            val_loss = []
            val_accuracy = []
            for i, (audio, masks, cap_inpt, cap_targ, cap_lens) in enumerate(
                val_loader
            ):
                audio, masks, cap_inpt, cap_targ = (
                    audio.to(device),
                    masks.to(device),
                    cap_inpt.to(device),
                    cap_targ.to(device),
                )

                out = model(audio, masks, cap_inpt)
                pred = out.logits
                cropped_pred = torch.cat(
                    [x[: cap_lens[i]] for i, x in enumerate(pred)], dim=0, device=device
                )

                loss = criterion(cropped_pred, cap_targ)

                precision = sum(torch.argmax(cropped_pred, dim=-1) == cap_targ) / len(
                    cap_targ
                )
                val_loss.append(loss.item())
                val_accuracy.append(precision)
        wandb.log(
            {
                "loss": running_loss / len(running_loss),
                "precision": sum(running_accuracy) / len(running_accuracy),
                "val_loss": sum(val_loss) / len(val_loss),
                "vali_precision": sum(val_accuracy) / len(val_accuracy),
            }
        )
        running_accuracy = []
        running_loss = []
        val_accuracy = []
        val_loss = []
