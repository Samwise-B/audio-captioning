# Audio Speaker Diarization Using WHISPER

This project showcases the development of a speaker diarization system by fine-tuning OpenAI's WHISPER model. The project involved extending WHISPER's tokenization capabilities with new speaker-specific tokens, enabling the model to identify and distinguish speakers in audio streams. By combining transcription with speaker recognition, the system demonstrates advanced audio processing capabilities.

## Key Features

### Machine Learning and Model Fine-Tuning
- **WHISPER Fine-Tuning**: Adapted WHISPER's capabilities to include speaker diarization by extending its tokenizer with new speaker tokens.
- **Multi-Task Learning**: Integrated transcription and speaker identification tasks into a single model pipeline.
- **Performance Optimization**: Tracked and analyzed training performance against a validation set to iteratively improve the model while minimizing overfitting.

### Tokenizer Customization
- **Speaker Tokens**: Extended the WHISPER tokenizer with speaker-specific tokens, a novel approach to enabling diarization.
- **Model Insights**: Identified that limiting the tokenizer to speaker tokens rather than full transcription could improve diarization performance.

### Training and Evaluation
- **Dataset Utilization**: Leveraged publicly available audio datasets (AMI) to train and evaluate the diarization system.

## Skills Demonstrated

### Technical Skills
- **Deep Learning**: Proficient in fine-tuning transformer models for novel tasks.
- **Audio Processing**: Experience in handling and processing audio data for machine learning applications.
- **Tokenizer Engineering**: Customized tokenization processes to align with task-specific requirements.
- **Framework Expertise**: Utilized frameworks like PyTorch, HuggingFace, and WANDB for model training and performance tracking.

### Problem-Solving Skills
- **System Optimization**: Identified trade-offs between transcription and diarization performance, providing actionable insights for future improvements.
- **Task Adaptation**: Adapted a pre-trained model to solve a complex, domain-specific problem effectively.

## Challenges and Lessons Learned
- **Tokenizer Design**: Balancing the dual tasks of transcription and diarization within the same model highlighted the importance of task-specific tokenization.
- **Data Constraints**: Ensured the model's robustness despite limited task-specific datasets by carefully designing training strategies.
- **Performance Bottlenecks**: Recognized that separating transcription and speaker diarization tasks might yield better overall performance in future iterations.

## Conclusion
This project is a testament to my ability to adapt state-of-the-art machine learning models to novel tasks, develop custom solutions for challenging problems, and deliver actionable insights for future improvements. It reflects my expertise in audio processing, tokenizer customization, and model evaluation, making me well-suited for roles in AI research, development, and deployment.

---

For further details or to discuss the project, feel free to reach out.
