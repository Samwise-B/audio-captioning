import torch
from torch.utils.data import DataLoader
import wandb

from data.homegrown import HomegrownDataset
from models.utils import validate_batch
from models.whisper import CustomWhisper
from training.utils import collate_fn

from backend.minio_client import MinioClientWrapper

def train(model, train_dataloader, val_dataloader, tokenizer, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for name,param in model.named_parameters():
        if(not param.requires_grad):
            print(f"Parameter {name} has requires_grad=False")
    
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for _, batch in enumerate(train_dataloader):
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
            
            loss = criterion(outputs.logits.transpose(1, 2), target_ids)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            wandb.log({"loss": loss.item(), "global_step": global_step})
            global_step += 1

        avg_loss = total_loss / len(train_dataloader)
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch})

        model.eval()
        with torch.no_grad():
            for batch_idx, val_batch in enumerate(val_dataloader):
                results = validate_batch(model, val_batch['audio'], val_batch['texts'], tokenizer)
                wandb.log({
                    f"validation_sample_{batch_idx}": results[0],  # Log each batch separately
                    "epoch": epoch
                })

def main():
    wandb.init(project="whisper-diarization", name="training_run")

    custom_model_wrapper = CustomWhisper(base_model="openai/whisper-tiny", max_speakers=5)
    tokenizer = custom_model_wrapper.tokenizer
    model = custom_model_wrapper.model

    minio = MinioClientWrapper()

    train_dataset = HomegrownDataset(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=10, collate_fn=collate_fn)

    val_dataset = HomegrownDataset(split='validate')
    val_dataloader = DataLoader(val_dataset, batch_size=3, collate_fn=collate_fn)

    train(model, train_dataloader, val_dataloader, tokenizer)

    local_weights_path = "./weights/whisper_diarization_v3.pth"
    torch.save(model.state_dict(), "./weights/whisper_diarization_v3.pth")
    minio.save_weights(local_weights_path, "whisper_diarization", "v3")

if __name__ == "__main__":
    main()