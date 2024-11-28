import whisper

import torch
from torch.utils.data import DataLoader

from data.homegrown import HomegrownDataset
from training.utils import add_speaker_tokens_to_whisper, clean_prediction, get_input_tensor, get_start_input_ids, get_target_input_ids

processor, model = add_speaker_tokens_to_whisper()
tokenizer = processor.tokenizer

eot_token = '<|endoftranscript|>'
eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)

dataset = HomegrownDataset()
dataloader = DataLoader(dataset, batch_size=1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

notimestamps_token_id = tokenizer.convert_tokens_to_ids('<|notimestamps|>')

def validate_sam(model, input_mel):
    model.eval()
    with torch.no_grad():
        input_tkns = get_start_input_ids(tokenizer)
        input_tkns_tensor = get_input_tensor(input_tkns, model.device)

        for i in range(80):
            initial_predictions = model(decoder_input_ids=input_tkns_tensor, input_features=input_mel)
            # 
            next_tkn = torch.argmax(initial_predictions.logits, dim=-1)[0,-1].unsqueeze(0)
            input_tkns_tensor = torch.cat((input_tkns_tensor.squeeze(), next_tkn), dim=0).unsqueeze(0)
            if input_tkns_tensor[-1, -1].item() == eot_token_id:
                break

    decoded_initial_output = tokenizer.decode(input_tkns_tensor.squeeze().tolist())
    print("===================")
    print("Initial Prediction:", decoded_initial_output)
    print("===================")

def train(model, input_mel, input_tensor, target_tensor):
    model.train()
    total_loss = 0
    for name,param in model.named_parameters():
        if(not param.requires_grad):
            print(f"Parameter {name} has requires_grad=False")
    
    for step in range(10):
        optimizer.zero_grad()
        
        outputs = model(
            decoder_input_ids=input_tensor, 
            input_features=input_mel
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



for batch in dataloader:
    # squeeze to remove channel dimension
    audio = batch['audio'].squeeze(1)
    # TODO is there a transformers method for this?
    input_mel = whisper.log_mel_spectrogram(audio).to(model.device)

    input_tkns_batched = [get_target_input_ids(transcript, tokenizer) for transcript in batch['transcript']]
    input_tkns_tensor = get_input_tensor(input_tkns_batched, model.device).squeeze(1)
    target_tkns_tensor = get_input_tensor(input_tkns_batched, model.device).squeeze(1)

    # shifting for teacher forcing
    input_tkns_tensor = input_tkns_tensor[:, :-1].contiguous()
    target_tkns_tensor = target_tkns_tensor[:, 1:].contiguous()


    train(model, input_mel, input_tkns_tensor, target_tkns_tensor)
    validate_sam(model, input_mel)
