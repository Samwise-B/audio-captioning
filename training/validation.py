

from models.utils import infer
from training.diarization_accuracy import calculate_der, calculate_der_numbered, parse_transcript, parse_transcript_numbered
from training.utils import clean_prediction
from jiwer import wer


def validate_batch(model, audio, targets, tokenizer, numbered_speakers=True):
    model.eval()
    batch_size = audio.shape[0]
    total_der = 0
    predictions = []
    for batch_idx in range(batch_size):

        audio_i = audio[batch_idx:batch_idx+1]
        target = targets[batch_idx:batch_idx+1][0]
        prediction = infer(model, audio_i, tokenizer)
        prediction = clean_prediction(prediction)
        predictions.append(prediction)

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
    avg_wer = wer(" ".join(targets), " ".join(predictions))
    avg_der = total_der / batch_size
    return avg_der, avg_wer