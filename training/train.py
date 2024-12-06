import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from data.ami import Ami
from models.CustomWhisper import CustomWhisper

from backend.minio_client import MinioClientWrapper
from training.validation import validate_batch
import tempfile

def compute_batch_loss(model, batch, device, criterion):
    audio = batch['input_features'].to(device)

    # Shift for teacher forcing
    input_ids = batch['input_ids'][:, :-1].contiguous().to(device)
    attention_mask = batch['attention_mask'][:, :-1].contiguous().to(device)
    target_ids = batch['input_ids'][:, 1:].clone().contiguous().to(device)

    outputs = model(
        audio,
        attention_mask,
        input_ids, 
    )

    loss = criterion(outputs.logits.transpose(1, 2), target_ids)
    return loss

def save_checkpoint(model, optimizer, epoch, global_step):
    """Saves model and optimizer state as a checkpoint directly to wandb."""    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, tmp_file.name)
        
        # Create and log artifact
        artifact = wandb.Artifact(
            name=f'model-checkpoint-{epoch}',
            type='model',
            description=f'Model checkpoint from epoch {epoch}, step {global_step}'
        )
        artifact.add_file(tmp_file.name)
        wandb.log_artifact(artifact)
        
    # Clean up temp file
    os.remove(tmp_file.name)
    
    print(f"Checkpoint saved to wandb at epoch {epoch}, step {global_step}")

def train(model, train_dataloader, val_dataloader, tokenizer, num_epochs=4, numbered_speakers=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            loss = compute_batch_loss(model, batch, device, criterion)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            wandb.log({"loss": loss.item(), "global_step": global_step})
            global_step += 1

        avg_loss = total_loss / len(train_dataloader)
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch})

        save_checkpoint(model, optimizer, epoch, global_step)

        model.eval()
        with torch.no_grad():
            total_der = 0
            total_val_loss = 0
            total_wer = 0
            for val_batch in tqdm(val_dataloader):
                der, wer = validate_batch(model, val_batch['input_features'], val_batch['text'], tokenizer, numbered_speakers=numbered_speakers)
                total_der += der
                total_wer += wer
                val_loss = compute_batch_loss(model, val_batch, device, criterion)
                total_val_loss += val_loss.item()
                wandb.log({"val_loss": val_loss.item(), "der": der, "wer": wer,  "global_step": global_step})
        
            wandb.log({
                "epocj_der": total_der / len(val_dataloader),
                "epoch_wer": total_wer / len(val_dataloader),
                "epoch_val_loss": total_val_loss / len(val_dataloader),
                "epoch": epoch
            })
        

def main():
    numbered_speakers=False
    wandb.init(project="whisper-diarization", name="training_run")

    custom_model_wrapper = CustomWhisper(base_model="openai/whisper-tiny", max_speakers=5, numbered_speakers=numbered_speakers)
    tokenizer = custom_model_wrapper.tokenizer
    model = custom_model_wrapper.model

    # minio = MinioClientWrapper()
    print("datasets and dataloaders")
    train_dataset = Ami(split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=Ami.get_collate_fn(train_dataset.tk, train_dataset.extractor))

    val_dataset = Ami(split="validation", subset_size=100)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=Ami.get_collate_fn(val_dataset.tk, val_dataset.extractor))
    print("finished datasets and dataloaders")

    train(model, train_dataloader, val_dataloader, tokenizer, num_epochs=4)

    torch.save(model.state_dict(), "whisper_diarization_ami.pth")
    artifact = wandb.Artifact('whisper_model', type='model')
    artifact.add_file("whisper_diarization_ami.pth")
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()