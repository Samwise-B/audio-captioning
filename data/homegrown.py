import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
from pathlib import Path
from pydub import AudioSegment
import tempfile
import json
import os

from data.utils import process_transcript_json

class HomegrownDataset(Dataset):
    """
    PyTorch Dataset for speaker diarization training data with M4A support
    """
    def __init__(
                self, 
                sample_rate=16000,
                duration=30,  # max duration in seconds
                transform=None,
                split='train',
                numbered_speakers=True
            ):
        """
        Args:
            audio_dir (str): Directory with all the M4A files
            sample_rate (int): Target sample rate for all audio
            duration (int): Target duration in seconds (will pad/trim)
            transform (callable, optional): Optional transform to be applied on audio
        """
        if(split == 'train'):
            audio_dir = os.path.join(os.getcwd(), "wav_files_train")
            self.transcript_filepath = 'training_data/training_data.json'
        if(split == 'validate'):
            audio_dir = os.path.join(os.getcwd(), "wav_files_validate")
            self.transcript_filepath = 'training_data/training_data_validation.json'
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform
        self.numbered_speakers = numbered_speakers
        
        # Get all m4a files and create metadata DataFrame
        self.files = list(self.audio_dir.glob('*.m4a'))
        self.metadata = self._create_metadata()
        
        # Create resampler if needed
        self.resampler = None
    
    def _create_metadata(self):
        """Create metadata DataFrame for all audio files"""
        data = []
        
        for audio_file in self.files:
            # Load m4a file to get properties
            audio = AudioSegment.from_file(audio_file, format="m4a")
            
            # Extract script number
            script_num = ''.join(filter(str.isdigit, audio_file.stem))
            
            data.append({
                'file_path': str(audio_file),
                'script_number': script_num,
                'duration': len(audio) / 1000.0,  # Convert ms to seconds
                'sample_rate': audio.frame_rate,
                'channels': audio.channels
            })
        
        return pd.DataFrame(data).sort_values('script_number')
    
    def _m4a_to_tensor(self, file_path):
        """Convert M4A file to audio tensor"""
        # Load M4A file
        audio = AudioSegment.from_file(file_path, format="m4a")
        
        # Export to WAV in memory
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_wav:
            audio.export(temp_wav.name, format='wav')
            # Load as tensor
            waveform, sr = torchaudio.load(temp_wav.name)
        
        return waveform, sr
    
    def _process_audio(self, audio, sr):
        """Process audio to target sample rate and duration"""
        # Resample if needed
        if sr != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sample_rate
                )
            audio = self.resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Calculate target length in samples
        target_length = int(self.sample_rate * self.duration)
        current_length = audio.shape[1]
        
        # Pad or trim to target length
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            audio = torch.nn.functional.pad(audio, (0, padding))
        else:
            # Trim to target length
            audio = audio[:, :target_length]
        
        return audio
    
    def _get_transcript_w_speaker_annotations(self, filename):
        with open(self.transcript_filepath, 'r') as f:
            data = json.load(f)
            processed_data = process_transcript_json(data, filename, numbered_speakers=self.numbered_speakers)
        return processed_data

    def __len__(self):
        """Return the number of audio files in the dataset"""
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get audio item by index
        
        Returns:
            dict: Contains audio tensor and metadata
        """
        print("getting item in homegrown dataset")
        # Get file path and metadata
        row = self.metadata.iloc[idx]
        file_path = row['file_path']
        
        # Load and convert M4A to tensor
        audio, sr = self._m4a_to_tensor(file_path)
        
        # Process audio
        audio = self._process_audio(audio, sr)
        filename = file_path.split('/').pop()
        target_transcript = self._get_transcript_w_speaker_annotations(filename)
        
        # Apply transforms if any
        if self.transform:
            audio = self.transform(audio)

        print("about to return from homegrown dataset")
        
        return {
            'audio': audio,
            'script_number': int(row['script_number']),
            'file_path': file_path,
            'duration': row['duration'],
            'transcript': target_transcript
        }

# Example usage
# if __name__ == "__main__":
#     # Create dataset

#     dataset = HomegrownDataset(
#         sample_rate=16000,
#         duration=30,
#         split='train',
#     )
    
#     # Create dataloader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=2,
#         shuffle=True,
#         num_workers=1,
#         collate_fn=collate_fn
#     )
    
#     # Print dataset info
#     print(f"\nDataset size: {len(dataset)}")
#     print("\nMetadata summary:")
#     print(dataset.metadata.describe())
    
#     # Example of loading a batch
#     for batch in dataloader:
#         print("\nBatch info:")
#         print(f"Audio shape: {batch['audio'].shape}")
#         print(f"Script numbers: {batch['script_number']}")
#         print(f"File path: {batch['file_path']}")
#         print(f"Transcript: {batch['transcript']}")
#         break