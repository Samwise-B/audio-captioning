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

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

for batch in dataloader:
    # squeeze to remove channel dimension
    audio = batch['audio'].squeeze(1)
    # TODO is there a transformers method for this?
    input_mel = whisper.log_mel_spectrogram(audio)
    input_tkns = get_start_input_ids()
    input_tkns_tensor = get_input_tensor(input_tkns)

    input_tkns_batched = [get_target_input_ids(transcript) for transcript in batch['transcript']]
    target_tkns_batched = [batch[1:] + [eot_token_id] for batch in input_tkns_batched]
    input_tkns_tensor = get_input_tensor(input_tkns_batched).squeeze(1)
    target_tkns_tensor = get_input_tensor(input_tkns_tensor).squeeze(1)
    torch.set_grad_enabled(True)
    model.train()
    for step in range(20):
        predictions = model(decoder_input_ids=input_tkns_tensor, input_features=input_mel).logits
        # transpose to get N,C,d (batch_size, vocab_size, seq_length). See pytorch docs
        # alternative is to use predictions.view(-1, predictions.shape[-1]) to get N,C (seq_length, vocab_size). Note that it's ok to flatten batch and seq length into one
        loss = criterion(predictions.transpose(1, 2), target_tkns_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step {step}, Loss: {loss.item()}")

    model.eval()
    torch.set_grad_enabled(False)
    # predicted_ids = model.generate(input_mel)
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
    # print(f"final_prediction: {transcription}")

    input_tkns = get_start_input_ids()
    input_tkns_tensor = get_input_tensor(input_tkns)
    for i in range(50):
        initial_predictions = model(decoder_input_ids=input_tkns_tensor, input_features=input_mel)
        next_tkn = torch.argmax(initial_predictions.logits, dim=-1)[0,-1].unsqueeze(0)
        input_tkns_tensor = torch.cat((input_tkns_tensor.squeeze(), next_tkn), dim=0).unsqueeze(0)
        if input_tkns_tensor[-1, -1].item() == eot_token_id:
                break

    decoded_initial_output = tokenizer.decode(input_tkns_tensor.squeeze().tolist())

    print("Final Prediction:", clean_prediction(decoded_initial_output))