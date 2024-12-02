import torch
from torch.utils.data import DataLoader

from data.homegrown import HomegrownDataset
from models.utils import validate_batch
from models.whisper import CustomWhisper
from training.utils import collate_fn

from backend.minio_client import MinioClientWrapper

def train(model, input_mel, input_tensor, target_tensor, mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = 0
    for name,param in model.named_parameters():
        if(not param.requires_grad):
            print(f"Parameter {name} has requires_grad=False")
    
    for step in range(10):
        optimizer.zero_grad()
        
        outputs = model(
            input_mel,
            mask,
            input_tensor, 
        )
        
        # TODO shift predictions here rather than before
        # shift_logits = outputs.logits[..., :-1, :].contiguous()
        # shift_targets = target_tensor[..., 1:].contiguous()
        # print(f"len: {len(model.parameters())}")
        
        loss = criterion(outputs.logits.transpose(1, 2), target_tensor)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Step {step}, Loss: {loss.item()}")

def main():
    custom_model_wrapper = CustomWhisper(base_model="openai/whisper-tiny", max_speakers=5)
    tokenizer = custom_model_wrapper.tokenizer
    model = custom_model_wrapper.model

    minio = MinioClientWrapper()
    dataset = HomegrownDataset(split='train')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    for i, batch in enumerate(dataloader):
        audio = batch['audio']
        targets = batch['texts']

        # shift for teacher forcing
        input_ids = batch['input_ids'][:, :-1].contiguous()
        attention_mask = batch['attention_mask'][:, :-1].contiguous()
        target_ids = batch['input_ids'][:, 1:].clone().contiguous()

        train(model, audio, input_ids, target_ids, attention_mask)
        validate_batch(model, audio, targets, tokenizer)
    local_weights_path = "./weights/whisper_diarization_v3.pth"
    torch.save(model.state_dict(), "./weights/whisper_diarization_v3.pth")
    minio.save_weights(local_weights_path, "whisper_diarization", "v3")

if __name__ == "__main__":
    main()