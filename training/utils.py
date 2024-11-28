import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def get_start_input_ids(tokenizer):
    input_ids = []

    sot_token = '<|startoftranscript|>'
    sot_token_id = tokenizer.convert_tokens_to_ids(sot_token)

    input_ids += [sot_token_id]

    language_token = '<|en|>'
    language_token_id = tokenizer.convert_tokens_to_ids(language_token)

    input_ids += [language_token_id]

    no_timestamps_token = '<|notimestamps|>'
    no_timestamps_token_id = tokenizer.convert_tokens_to_ids(no_timestamps_token)
    input_ids += [no_timestamps_token_id]
    return input_ids

# def get_input_tensor(input_ids, device):
#     input_tkns = torch.tensor(input_ids).unsqueeze(0).to(device)
#     return input_tkns

def get_target_input_ids(transcript, tokenizer):
    input_ids = get_start_input_ids(tokenizer)
    input_ids += tokenizer.encode(transcript, add_special_tokens=False)
    return input_ids

def add_speaker_tokens_to_whisper(model_name="openai/whisper-tiny", speaker_labels=None):
    """
    Add custom speaker tokens to a Whisper model's tokenizer and processor
    """
    if speaker_labels is None:
        speaker_labels = [f"<|speaker_{i}|>" for i in range(1, 5)]
    # Initialize components
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Get the current vocab size
    original_vocab_size = len(processor.tokenizer)
    
    # Add new tokens to both the processor's tokenizer and the decoder
    num_added_tokens = processor.tokenizer.add_tokens(speaker_labels, special_tokens=True)
    processor.feature_extractor.tokenizer = processor.tokenizer  # Ensure feature_extractor uses updated tokenizer
    
    if num_added_tokens > 0:
        # Resize the token embeddings matrix of the model
        model.resize_token_embeddings(len(processor.tokenizer))
        
        # Initialize the new embeddings
        with torch.no_grad():
            existing_embeddings = model.get_input_embeddings().weight[:original_vocab_size]
            mean_embedding = torch.mean(existing_embeddings, dim=0)
            
            for i in range(num_added_tokens):
                new_token_idx = original_vocab_size + i
                model.get_input_embeddings().weight[new_token_idx] = mean_embedding.clone()
    
    return processor, model


def clean_prediction(decoded_text):
    special_tokens = ["<|en|>", "<|transcribe|>", "<|notimestamps|>", "<|endoftext|>"]
    for token in special_tokens:
        decoded_text = decoded_text.replace(token, "").strip()
    return decoded_text