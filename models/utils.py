import torch

def get_start_input_ids(tokenizer):
    input_ids = []

    sot_token = '<|startoftranscript|>'
    sot_token_id = tokenizer.convert_tokens_to_ids(sot_token)

    input_ids += [sot_token_id]

    no_timestamps_token = '<|notimestamps|>'
    no_timestamps_token_id = tokenizer.convert_tokens_to_ids(no_timestamps_token)
    input_ids += [no_timestamps_token_id]
    return input_ids

def clean_prediction(decoded_text):
    special_tokens = [
        "<|en|>",
        "<|transcribe|>",
        # TODO why are we ever returning <|translate|>??
        "<|translate|>",
        "<|notimestamps|>",
        "<|endoftext|>",
        "<|startoftranscript|>",
    ]
    for token in special_tokens:
        decoded_text = decoded_text.replace(token, "").strip()
    return decoded_text

def infer(model, input_mel, tokenizer):
    eot_token = '<|endoftranscript|>'
    eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)
    model.eval()
    with torch.no_grad():
        #  this is different to how I tokenize in the collate_fn
        input_ids = get_start_input_ids(tokenizer) # 50258, 50363 ('<|startoftranscript|>', '<|notimestamps|>')
        input_tkns_tensor = torch.tensor(input_ids).unsqueeze(0).to(model.device)

        for i in range(80):
            initial_predictions = model(decoder_input_ids=input_tkns_tensor, input_features=input_mel)
            next_tkn = torch.argmax(initial_predictions.logits, dim=-1)[0,-1].unsqueeze(0)
            input_tkns_tensor = torch.cat((input_tkns_tensor.squeeze(), next_tkn), dim=0).unsqueeze(0)
            if input_tkns_tensor[-1, -1].item() == eot_token_id:
                break

    decoded_initial_output = tokenizer.decode(input_tkns_tensor.squeeze().tolist())
    return decoded_initial_output
