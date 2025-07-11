# Real-Time Speech Style Transfer

A modular deep learning system for real-time speech style transfer that modifies the tone, emotion, or accent of speech while preserving its linguistic content and speaker identity.

---

## 🚀 Features
- **Cascaded Architecture**: Content encoder, speaker encoder, style modulator, and vocoder
- **Disentangled Representation Learning**: Modular design separates content, identity, and emotion
- **Real-Time Ready**: Sub-500ms inference pipeline using HiFi-GAN and Wav2Vec 2.0
- **Evaluation Suite**: Spectrogram visualizer, cosine similarity, and ABX-style metrics

---

## 📁 Project Structure
```
├── train.py                    # Full training pipeline
├── inference.py                # Style transfer using trained models
├── evaluate.py                 # Spectrogram and metric visualization
├── models/
│   ├── content_encoder.py      # Wav2Vec2-based content extractor
│   ├── speaker_encoder.py      # Triplet-loss-based identity encoder
│   ├── style_encoder.py        # Prosody and emotion extractor
│   ├── style_modulator.py      # Fusion and transformation module
│   └── vocoder.py              # HiFi-GAN audio synthesis
├── utils/                      # Preprocessing, augmentation
├── data/                       # Preprocessed dataset (CREMA-D)
├── output.wav                  # Stylized output example
├── spectrogram_comparison.png # Visual result
```

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

---

## 👥 Authors
- Pasupuleti Jaswanth
- Shamil Saidu
- Harini CJ
- Ms Kavya Sai

