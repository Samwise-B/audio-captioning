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

def validate_batch(model, audio, targets, tokenizer, numbered_speakers=True):
    model.eval()
    batch_size = audio.shape[0]
    total_der = 0
    for batch_idx in range(batch_size):
        audio_i = audio[batch_idx:batch_idx+1]
        target = targets[batch_idx:batch_idx+1][0]
        prediction = infer(model, audio_i, tokenizer)
        prediction = clean_prediction(prediction)
        
        print("===================")
        print(f"Batch item {batch_idx}\n")
        print(f"Target:\n")
        print(target)
        print(f"\nPrediction:\n")
        print(prediction)

        if numbered_speakers:
            output_utterances = parse_transcript_numbered(prediction)
            target_utterances = parse_transcript_numbered(target)
            der = calculate_der_numbered(output_utterances, target_utterances)
            total_der += der
        else:
            output_utterances = parse_transcript(prediction)
            target_utterances = parse_transcript(target)
            der = calculate_der(output_utterances, target_utterances)
            total_der += der
        print(f"der:{der}")
        print("===================")
    avg_der = total_der / batch_size
    return avg_der