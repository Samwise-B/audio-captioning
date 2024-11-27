# from pydub import AudioSegment
# import os


# def split_audio(audio, chunk_length_ms=30000):
#     """
#     Splits an audio file into chunks of specified length.

#     Args:
#         file_path (str): Path to the .wav file.
#         chunk_length_ms (int): Length of each chunk in milliseconds (default is 30 seconds).

#     Returns:
#         List of paths to the generated audio chunks.
#     """
#     if audio.shape[0] <= 480000:
#         return [audio]

#     # Split the audio and save chunks
#     chunks = []
#     for i in range(0, audio.shape[0], chunk_length_ms):
#         chunk = audio[i : i + chunk_length_ms]
#         chunks.append(chunk)

#     print(f"Audio split into {len(chunks)} chunks.")
#     return chunks


# # Example usage
# file_path = "example.wav"  # Path to your .wav file
# split_audio(file_path)
