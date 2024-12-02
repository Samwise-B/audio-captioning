import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper

from models.whisper import CustomWhisper

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

def collate_fn(batch):
    # Handle audio padding
    audio_features = [whisper.log_mel_spectrogram(item['audio'].squeeze()) for item in batch]
    padded_audio = torch.nn.utils.rnn.pad_sequence(audio_features, batch_first=True)
    
    # Handle text padding using HF
    texts = [item['transcript'] for item in batch]
    custom_whisper = CustomWhisper(base_model="openai/whisper-tiny", max_speakers=5)
    tokenizer = custom_whisper.tokenizer
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    
    return {
        'audio': padded_audio,
        'input_ids': tokenized.input_ids,
        'attention_mask': tokenized.attention_mask,
        'texts': texts
    }

def get_target_input_ids(transcript, tokenizer):
    input_ids = get_start_input_ids(tokenizer)
    input_ids += tokenizer.encode(transcript, add_special_tokens=False)
    return input_ids

def clean_prediction(decoded_text):
    special_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>", "<|endoftext|>"]
    for token in special_tokens:
        decoded_text = decoded_text.replace(token, "").strip()
    return decoded_text