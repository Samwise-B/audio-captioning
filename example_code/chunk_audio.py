from pydub import AudioSegment
import os


def split_audio(audio, sample_rate=16000, chunk_size_sec=30, overlap=5):
    """
    Splits an audio file into chunks of specified length.

    Args:
        file_path (str): Path to the .wav file.
        chunk_length_ms (int): Length of each chunk in milliseconds (default is 30 seconds).

    Returns:
        List of paths to the generated audio chunks.
    """
    chunk_length = sample_rate * chunk_size_sec
    step_size = sample_rate * (chunk_size_sec - overlap)
    if audio.shape[0] <= chunk_length:
        return [audio]

    # Split the audio and save chunks
    chunks = []
    for i in range(0, audio.shape[0], step_size):
        chunk = audio[i : i + chunk_length]
        chunks.append(chunk)

    print(f"Audio split into {len(chunks)} chunks.")
    return chunks
