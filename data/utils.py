def process_transcript_json(data_array, filename):
    file_data = next(item for item in data_array if item["audio_path"] == filename)
    transcript_with_speaker_annotations = ''
    for i, snippet in enumerate(file_data["speaker_turns"]):
        transcript_with_speaker_annotations += f" <|speaker_{snippet["speaker"]}|> "
        transcript_with_speaker_annotations += f"{snippet['text']}"
    return transcript_with_speaker_annotations