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


def validate(model, audio):
    model.eval()
    torch.set_grad_enabled(False)
    
    inputs = processor(audio.squeeze(0), sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    generated_ids = model.generate(
        inputs["input_features"],
        attention_mask=inputs["attention_mask"]
    )
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=False)
    print(f"final_prediction from generate: {transcription}")

def validate_sam(model, input_mel):
    model.eval()
    torch.set_grad_enabled(False)
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
    torch.set_grad_enabled(True)
    model.train()
    for step in range(10):
        predictions = model(decoder_input_ids=input_tensor, input_features=input_mel).logits
        loss = criterion(predictions.transpose(1, 2), target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step {step}, Loss: {loss.item()}")


for batch in dataloader:
    # squeeze to remove channel dimension
    audio = batch['audio'].squeeze(1)
    # TODO is there a transformers method for this?
    input_mel = whisper.log_mel_spectrogram(audio)

    input_tkns_batched = [get_target_input_ids(transcript, tokenizer) for transcript in batch['transcript']]
    target_tkns_batched = [batch[1:] + [eot_token_id] for batch in input_tkns_batched]
    input_tkns_tensor = get_input_tensor(input_tkns_batched, model.device).squeeze(1)
    target_tkns_tensor = get_input_tensor(input_tkns_tensor, model.device).squeeze(1)

    train(model, input_mel, input_tkns_tensor, target_tkns_tensor)
    validate_sam(model, input_mel)
