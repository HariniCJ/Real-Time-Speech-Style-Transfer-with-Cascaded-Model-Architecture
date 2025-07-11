# 🎙️ Real-Time Speech Style Transfer with Cascaded Architecture

A deep learning-based system for **real-time speech style transfer** using a modular architecture that disentangles content, speaker identity, and prosody. The system leverages pre-trained models like **Wav2Vec 2.0** and **HiFi-GAN** to generate expressive, speaker-consistent speech outputs in new styles.

---

## 🚀 Key Features

- 🔊 **Real-time speech style transfer** pipeline
- 🧠 **Modular architecture**:
  - Content Encoder: `Wav2Vec2` (frozen)
  - Speaker & Style Encoders: Custom CNNs
  - Style Modulator: MLP-based fusion layer
  - Vocoder: `HiFi-GAN` (frozen)
- 🎯 Disentangles linguistic, speaker, and style representations
- 🔁 Training is done only on the **style modulator** and vocoder projection layer
- 📈 High cosine similarity scores on content (1.00), style (0.97), and speaker (0.98)

---

## 🧩 Project Structure
├── my_models/ # CNN architectures and modulator
├── my_modules/ # Core pipeline logic
├── my_utils/ # Audio I/O, transforms, spectrogram tools
├── speech_style_transfer_samples/ # RAVDESS audio samples (content/speaker/style)
├── config.py # Configuration constants
├── train.py # Training script for modulator + projection
├── inference.py # Inference pipeline with CLI
├── test.py # Evaluation scripts and metrics
├── requirements.txt # Python dependencies


---

## 📦 Installation
bash
git clone https://github.com/your-username/speech-style-transfer.git
cd speech-style-transfer
pip install -r requirements.txt

🎓 Dataset
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
1,440 high-quality WAV speech files across 8 emotions (with strong & normal intensity)
Balanced across 24 professional actors (12 male, 12 female)
Preprocessed into:
  -16kHz resampled waveform
  -Mel spectrograms for style/speaker encoders

🧠 Model Pipeline
[Input Speech] → Wav2Vec2 → Content Embeddings
[Speaker Ref]  → CNN Encoder → Speaker Embedding
[Style Ref]    → CNN Encoder → Style Embedding
→ Style Modulator (MLP) → Modulated Features → HiFi-GAN Vocoder → Output Speech
⚠️ Note: Current vocoder projection step is a limitation — HiFi-GAN input requires refinement for fully natural audio generation.

🧪 Evaluation Results
Metric	Score
🎯 Content Preservation	1.0000
🗣️ Speaker Similarity	0.9771
🎭 Style Transfer Score	0.9679

Qualitative and spectrogram analysis confirms accurate modulation of prosodic features and retention of speaker identity.

📉 Limitations
Current HiFi-GAN projection layer does not generate valid Mel spectrograms → waveform synthesis blocked.

Future work:
  -Fine-tune vocoder on modulated embeddings
  -Introduce prosody-aware loss & energy normalization
  -Add perceptual metrics (e.g., MOS, ABX)
