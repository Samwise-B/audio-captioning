import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data.homegrown import HomegrownDataset
from models.CustomWhisper import CustomWhisper
from training.utils import collate_fn

from backend.minio_client import MinioClientWrapper
from training.validation import validate_batch

def save_checkpoint(model, optimizer, epoch, global_step, checkpoint_dir="./checkpoints"):
    """Saves model and optimizer state as a checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{global_step}.pth")
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def train(model, train_dataloader, val_dataloader, tokenizer, num_epochs=10, numbered_speakers=True, checkpoint_dir="./checkpoints"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # for name,param in model.named_parameters():
    #     if(param.requires_grad):
    #         print(f"Parameter {name} has requires_grad=True")
    #     if( not param.requires_grad):
    #         print(f"Parameter {name} has requires_grad=False")
    
    
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for _, batch in tqdm(enumerate(train_dataloader)):
            print(f"batch")
            audio = batch['audio']

            # shift for teacher forcing
            input_ids = batch['input_ids'][:, :-1].contiguous()
            attention_mask = batch['attention_mask'][:, :-1].contiguous()
            target_ids = batch['input_ids'][:, 1:].clone().contiguous()

            optimizer.zero_grad()
            
            outputs = model(
                audio,
                attention_mask,
                input_ids, 
            )
            print(f"output")
            
            loss = criterion(outputs.logits.transpose(1, 2), target_ids)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            wandb.log({"loss": loss.item(), "global_step": global_step})
            global_step += 1

        avg_loss = total_loss / len(train_dataloader)
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch})

        save_checkpoint(model, optimizer, epoch, global_step, checkpoint_dir)

        model.eval()
        with torch.no_grad():
            for _, val_batch in tqdm(enumerate(val_dataloader)):
                avg_der = validate_batch(model, val_batch['audio'], val_batch['texts'], tokenizer, numbered_speakers=numbered_speakers)
                wandb.log({
                    "avg_der": avg_der,
                    "epoch": epoch
                })

def main():
    numbered_speakers=False
    wandb.init(project="whisper-diarization", name="training_run")

    custom_model_wrapper = CustomWhisper(base_model="openai/whisper-tiny", max_speakers=5, numbered_speakers=False)
    tokenizer = custom_model_wrapper.tokenizer
    model = custom_model_wrapper.model

    # minio = MinioClientWrapper()
    print("datasets and dataloaders")
    train_dataset = HomegrownDataset(split='train', numbered_speakers=numbered_speakers)
    train_dataloader = DataLoader(train_dataset, batch_size=10, collate_fn=collate_fn)

    val_dataset = HomegrownDataset(split='validate', numbered_speakers=numbered_speakers)
    val_dataloader = DataLoader(val_dataset, batch_size=3, collate_fn=collate_fn)
    print("finished datasets and dataloaders")

    train(model, train_dataloader, val_dataloader, tokenizer, numbered_speakers=numbered_speakers)

    local_weights_path = "./weights/whisper_diarization_v3.pth"
    torch.save(model.state_dict(), "./weights/whisper_diarization_v3.pth")
    # minio.save_weights(local_weights_path, "whisper_diarization", "v3")

if __name__ == "__main__":
    main()