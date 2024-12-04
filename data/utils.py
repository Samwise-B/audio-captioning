def process_transcript_json(data_array, filename, numbered_speakers=True):
    file_data = next(item for item in data_array if item["audio_path"] == filename)
    transcript_with_speaker_annotations = ''
    for i, snippet in enumerate(file_data["speaker_turns"]):
        if i > 0:
            transcript_with_speaker_annotations += ' '
        if numbered_speakers:
            transcript_with_speaker_annotations += f"<|speaker_{(snippet['speaker'])}|> "
        elif i > 0:
            transcript_with_speaker_annotations += f"<|speaker_change|> "
        transcript_with_speaker_annotations += f"{snippet['text']}"
    return transcript_with_speaker_annotations
