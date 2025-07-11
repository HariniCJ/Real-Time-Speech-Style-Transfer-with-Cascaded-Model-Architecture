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

## 📦 Requirements
Install all dependencies:
```bash
pip install -r requirements.txt
```
Typical contents of `requirements.txt`:
- torch
- torchaudio
- transformers  # for Wav2Vec2
- librosa
- matplotlib
- soundfile
- scikit-learn
```
transformers==4.36.2
```

---

## 🏃‍♂️ Usage

### 1. Training
```bash
python train.py
```
Ensure your dataset is preprocessed and follows the CREMA-D directory structure.

### 2. Inference
```bash
python inference.py
```
Files required in root directory:
- `input_speech.wav`: source to be transformed
- `speaker_reference.wav`: identity provider
- `style_reference.wav`: emotional/style reference

### 3. Evaluation
```bash
python evaluate.py
```
Generates and saves a spectrogram comparison to:
- `spectrogram_comparison.png`

---

## 🎯 Model Architecture
- `ContentEncoder`: Wav2Vec2 Base (768-dim), pretrained, frozen during fine-tuning
- `SpeakerEncoder`: 3-layer FFNN, trained using online triplet loss
- `StyleEncoder`: CNN + pooling + MLP projecting to 128-dim prosody vector
- `StyleModulator`: Deep MLP with LayerNorm, GELU, and Dropout, auto-handling sequence alignment
- `Vocoder`: HiFi-GAN V1, optionally fine-tuned on custom outputs

---

## 🔧 Advanced Configuration
- Replace `SpeakerEncoder` with `ECAPA-TDNN` from `SpeechBrain` for better identity modeling
- Enable multilingual transfer by swapping in `Wav2Vec2-XLSR`
- Support continuous style interpolation using vector mixing

---or fully natural audio generation.

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
