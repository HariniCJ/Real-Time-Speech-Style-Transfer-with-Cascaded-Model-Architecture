
import os
import zipfile

import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor


def download_ravdess():

    if not os.path.exists("RAVDESS"):
        print("Downloading RAVDESS dataset...")
        os.system("kaggle datasets download -d jeffreybraun/ravdess-emotional-speech-audio")
        with zipfile.ZipFile("ravdess-emotional-speech-audio.zip", 'r') as zip_ref:
            zip_ref.extractall("RAVDESS")
        print("Download and extraction complete.")
    else:
        print("RAVDESS dataset already exists.")

def preprocess_audio(filepath, target_sr=16000):
    waveform, sr = torchaudio.load(filepath)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    waveform = waveform / waveform.abs().max()
    return waveform

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80, f_min=0, f_max=8000)

def get_mel_spectrogram(waveform):
    mel = mel_transform(waveform)
    mel = torch.log(torch.clamp(mel, min=1e-9))
    return mel

def load_hifigan():
    hifigan = torchaudio.pipelines.HIFI_GAN_V1.get_model()
    hifigan.eval()
    return hifigan

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def extract_content_embedding(waveform, sr=16000):
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def main():
    download_ravdess()
    example_file = "RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
    if not os.path.exists(example_file):
        print(f"File not found: {example_file}")
        return

    waveform = preprocess_audio(example_file)
    mel = get_mel_spectrogram(waveform)

    hifigan = load_hifigan()
    with torch.no_grad():
        audio_out = hifigan(mel.unsqueeze(0)).squeeze().cpu()
    sf.write("output_hifigan.wav", audio_out.numpy(), 16000)
    print("Generated audio saved as output_hifigan.wav")

    content_emb = extract_content_embedding(waveform)
    print(f"Content embedding shape: {content_emb.shape}")

if __name__ == "__main__":
    main()
