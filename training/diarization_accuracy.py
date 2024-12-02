
from dataclasses import dataclass
import re
from typing import List


@dataclass
class Utterance:
    speaker: str
    text: str

def parse_transcript(transcript: str) -> List[Utterance]:
    """Parse the transcript format into a list of Utterances"""
    # Remove transcript markers and language tags
    cleaned = transcript.replace("<|startoftranscript|>", "").replace("<|en|>", "").replace("<|notimestamps|>", "").replace("<|endoftext|>", "")
    
    # Split into utterances and parse
    utterances = []
    pattern = r"<\|speaker_(\d+)\|>\s*(.*?)(?=<\|speaker_\d+\|>|$)"
    matches = re.finditer(pattern, cleaned, re.DOTALL)
    
    for match in matches:
        speaker = f"speaker_{match.group(1)}"
        text = match.group(2).strip()
        utterances.append(Utterance(speaker=speaker, text=text))
    
    return utterances

def calculate_der(output: List[Utterance], target: List[Utterance]) -> float:
    """
    Calculate Diarization Error Rate (DER)
    DER = Number of speaker errors / Total number of utterances
    """
    speaker_errors = 0
    if len(output) != len(target):
        #  allow different number of speaker blocks, but penalise
        speaker_errors += abs(len(output) - len(target))
    
    speaker_errors += sum(1 for o, t in zip(output, target) if o.speaker != t.speaker)

    print(speaker_errors)
    return speaker_errors / len(target)


if __name__ == "__main__":
    # FROM CURRENT TRAINING IE. OVERFITTED
    output = """
    <|speaker_1|> I'd like to book a grooming appointment for my golden retriever
    <|speaker_2|> We have openings this Saturday morning. Has your dog been here before?
    <|speaker_1|> No, this would be his first time
    <|speaker_1|> The full grooming package for large dogs is $85. Would you like 9 AM or 11 AM?
    <|speaker_1|> 9 AM works better for me
    """

    target = """
    <|speaker_1|> I'd like to book a grooming appointment for my golden retriever
    <|speaker_2|> We have openings this Saturday morning. Has your dog been here before?
    <|speaker_1|> No, this would be his first time
    <|speaker_2|> The full grooming package for large dogs is $85. Would you like 9 AM or 11 AM?
    <|speaker_1|> 9 AM works better for me
    """



    # FROM CURRENT VALIDATION
    # output= """
    # <|speaker_1|> No
    # <|speaker_2|> I see, in the book I see, at this saying, your milestones are Saturday
    # <|speaker_2|> Really, I was glad it could go high calendar
    # <|speaker_2|> You might want to reschedule, the predicting circumstances
    # """

    # target = """
    # <|speaker_1|> Have you seen the forecast for this weekend?
    # <|speaker_2|> Yes, they're saying it might snow on Saturday.
    # <|speaker_1|> Really? I was planning to go hiking then.
    # <|speaker_2|> You might want to reschedule. They're predicting several inches
    # <|speaker_1|> Thanks for letting me know. I'll make other plans."
    # """

    output_utterances = parse_transcript(output)
    target_utterances = parse_transcript(target)
    der = calculate_der(output_utterances, target_utterances)
    print(f"der:{der}")
