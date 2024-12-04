import torch
import whisper

from models.CustomWhisper import CustomWhisper

def collate_fn(batch):
    # Handle audio padding
    audio_features = [whisper.log_mel_spectrogram(item['audio'].squeeze()) for item in batch]
    padded_audio = torch.nn.utils.rnn.pad_sequence(audio_features, batch_first=True)
    
    # Handle text padding using HF
    texts = [item['transcript'] for item in batch]
    custom_whisper = CustomWhisper(base_model="openai/whisper-tiny", max_speakers=5)
    tokenizer = custom_whisper.tokenizer
    tokenized = tokenizer(texts, padding=True, return_tensors="pt") # [50258, 50363, ...] : '<|startoftranscript|>''<|notimestamps|>' ..
    
    return {
        'audio': padded_audio,
        'input_ids': tokenized.input_ids,
        'attention_mask': tokenized.attention_mask,
        'texts': texts
    }

def clean_prediction(decoded_text):
    special_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>", "<|endoftext|>"]
    for token in special_tokens:
        decoded_text = decoded_text.replace(token, "").strip()
    return decoded_text
