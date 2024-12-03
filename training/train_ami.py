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
from models.CustomWhisper import CustomWhisper
from utils.inference import run_inference

model_dir = Path(__file__).parent.parent / "weights"

# Check if the directory exists
if not model_dir.exists():
    # Create the directory
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory created: {model_dir}")

torch.manual_seed(42)
logging.basicConfig(
    filename="training.log",  # File to log to
    filemode="w",  # Overwrite file each time; use "a" to append
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

subset_size = None
batch_size = 32
ds = Ami("train", subset_size=subset_size)
dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=Ami.collate_fn)

val_ds = Ami("validation", subset_size=subset_size)
val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=Ami.collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomWhisper()
model.to(device)
model_name = "whisper-diarization-full"

print(
    f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

val_freq = (ds.__len__() // batch_size) / 4
num_epochs = 1000

config = {
    "batch_size": batch_size,
    "subset_size": subset_size,
    "learning_rate": 1e-4,
    "num_epochs": num_epochs,
    "val_freq": val_freq,
}

wandb.init(project="audio-captioning", name=model_name, config=config)
running_loss = []
running_accuracy = []
for epoch in range(num_epochs):
    for idx, (audio, masks, cap_inpt, cap_targ, cap_lens) in enumerate(
        tqdm(dataloader, desc=f"Training {epoch}")
    ):
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
            [x[: cap_lens[i]] for i, x in enumerate(pred)], dim=0
        ).to(device)
        loss = criterion(cropped_pred, cap_targ)
        loss.backward()
        optimizer.step()

        precision = sum(torch.argmax(cropped_pred, dim=-1) == cap_targ) / len(cap_targ)
        running_loss.append(loss.item())
        running_accuracy.append(precision)
        # print(f"epoch: {epoch} loss: {loss.item()}, precision: {precision}", end="\r")

        if (idx + 1) % val_freq == 0:
            torch.save(
                model.state_dict(),
                model_dir / f"{model_name}.pt",
            )
            wandb.save(
                str(model_dir / f"{model_name}.pt"),
                base_path=str(model_dir),
            )
            logging.info("Running inference...")
            o, t = run_inference(model, ds, 0, 100)
            logging.info(f"Inference: \n{o}")
            logging.info("=======================")
            logging.info(f"Target: \n{t}")

            # clear training batch from memory
            del audio, masks, cap_inpt, cap_targ, cap_lens, pred, cropped_pred, loss
            torch.cuda.empty_cache()

            # validation loop
            with torch.inference_mode():
                val_loss = []
                val_accuracy = []
                for audio, masks, cap_inpt, cap_targ, cap_lens in tqdm(
                    val_loader, desc="Validation"
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
                        [x[: cap_lens[i]] for i, x in enumerate(pred)], dim=0
                    ).to(device)

                    loss = criterion(cropped_pred, cap_targ)

                    precision = sum(
                        torch.argmax(cropped_pred, dim=-1) == cap_targ
                    ) / len(cap_targ)
                    val_loss.append(loss.item())
                    val_accuracy.append(precision)
            wandb.log(
                {
                    "loss": sum(running_loss) / len(running_loss),
                    "precision": sum(running_accuracy) / len(running_accuracy),
                    "val_loss": sum(val_loss) / len(val_loss),
                    "vali_precision": sum(val_accuracy) / len(val_accuracy),
                }
            )
            running_accuracy = []
            running_loss = []
            val_accuracy = []
            val_loss = []

torch.save(
    model.state_dict(),
    model_dir / f"final_model.pt",
)
wandb.save(
    str(model_dir / f"final_model.pt"),
    base_path=str(model_dir),
)
wandb.finish()
