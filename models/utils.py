import torch

from training.diarization_accuracy import calculate_der, parse_transcript

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

def validate(model, input_mel, tokenizer):
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

def validate_batch(model, audio, targets, tokenizer):
    model.eval()
    batch_size = audio.shape[0]
    results = []
    total_der = 0
    for batch_idx in range(batch_size):
        audio_i = audio[batch_idx:batch_idx+1]
        target = targets[batch_idx:batch_idx+1][0]
        prediction = validate(model, audio_i, tokenizer)
        prediction = clean_prediction(prediction)
        
        result = {
            "target": target,  # Assuming target is a list/tensor
            "prediction": prediction
        }
        results.append(result)
        
        print("===================")
        print(f"Batch item {batch_idx}\n")
        print(f"Target:\n")
        print(target)
        print(f"\nPrediction:\n")
        print(prediction)

        output_utterances = parse_transcript(prediction)
        target_utterances = parse_transcript(target)
        der = calculate_der(output_utterances, target_utterances)
        total_der += der
        print(f"der:{der}")
        print("===================")
    avg_der = total_der / batch_size
    return avg_der