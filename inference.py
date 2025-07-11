import os
import torch
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import torchaudio
import matplotlib.pyplot as plt

from config import Config
from my_models.content_encoder import ContentEncoder
from my_models.speaker_encoder import SpeakerEncoder
from my_models.style_encoder import StyleEncoder
from my_modules.style_modulator import StyleModulator
from my_models.vocoder import HiFiGANVocoder
from my_utils.audio import load_audio

def extract_mel_spectrogram(audio, sr=16000, n_fft=1024, hop_length=256, n_mels=80, f_min=0.0, f_max=8000):
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    audio = audio.float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=f_min, f_max=f_max, power=2.0
    )
    mel_spectrogram = mel_transform(audio)
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    mel_db = db_transform(mel_spectrogram)
    return mel_db.float()

INPUT_DIR = "speech_style_transfer_samples/"
INPUT_PATH = os.path.join(INPUT_DIR, "input_speech.wav")
STYLE_PATH = os.path.join(INPUT_DIR, "style_reference.wav")
SPEAKER_PATH = os.path.join(INPUT_DIR, "speaker_reference.wav")
OUTPUT_PATH = "output.wav"
SPECTROGRAM_IMG_PATH = "spectrogram_comparison.png"
CHECKPOINT_PATH = os.path.join(INPUT_DIR, "speech_style_transfer_model.pth")

def preprocess_input(audio_path, target_sr=16000, audio_length=3):
    audio = load_audio(audio_path, sr=target_sr)
    audio = audio[: target_sr * audio_length]
    if len(audio) < target_sr * audio_length:
        padding = np.zeros(target_sr * audio_length - len(audio))
        audio = np.concatenate([audio, padding])
    audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-9)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float()
    return audio_tensor

def visualize_modulator_output(content, modulated, sample_idx=0, num_channels=5):
    content_np = content[sample_idx, :, :num_channels].cpu().detach().numpy()
    modulated_np = modulated[sample_idx, :, :num_channels].cpu().detach().numpy()
    T = content_np.shape[0]
    time = range(T)
    plt.figure(figsize=(12, 6))
    for ch in range(num_channels):
        plt.plot(time, content_np[:, ch], label=f'Content Ch {ch}')
        plt.plot(time, modulated_np[:, ch], linestyle='dashed', label=f'Modulated Ch {ch}')
    plt.xlabel("Time steps")
    plt.ylabel("Feature value")
    plt.title("Content vs Modulated Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()

def visualize_modulator_mel(content_mel, style_mel, modulated_mel):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    librosa.display.specshow(content_mel, sr=Config.SAMPLE_RATE, hop_length=256, y_axis='mel', x_axis='time', fmax=8000)
    plt.title("Content Mel")
    plt.colorbar(format='%+2.0f dB')
    plt.subplot(1, 3, 2)
    librosa.display.specshow(style_mel, sr=Config.SAMPLE_RATE, hop_length=256, y_axis='mel', x_axis='time', fmax=8000)
    plt.title("Style Mel")
    plt.colorbar(format='%+2.0f dB')
    plt.subplot(1, 3, 3)
    librosa.display.specshow(modulated_mel, sr=Config.SAMPLE_RATE, hop_length=256, y_axis='mel', x_axis='time', fmax=8000)
    plt.title("Modulated Mel (Output)")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig("modulator_mel_comparison.png")
    print("✅ Saved modulator mel spectrogram comparison to modulator_mel_comparison.png")
    plt.close()

def griffin_lim(mel_db, n_iter=32, sr=16000, n_fft=1024, hop_length=256):
    mel_power = librosa.db_to_power(mel_db)
    inv_mel = librosa.feature.inverse.mel_to_stft(mel_power, sr=sr, n_fft=n_fft, fmax=8000)
    audio = librosa.griffinlim(inv_mel, n_iter=n_iter, hop_length=hop_length, win_length=n_fft)
    return audio

def run_inference():
    content_encoder = ContentEncoder(Config.CONTENT_ENCODER_OUTPUT_DIM).to(Config.DEVICE)
    speaker_encoder = SpeakerEncoder(80, Config.SPEAKER_ENCODER_EMBED_DIM).to(Config.DEVICE)
    style_encoder = StyleEncoder(80, Config.STYLE_ENCODER_EMBED_DIM).to(Config.DEVICE)
    style_modulator = StyleModulator(
        Config.CONTENT_ENCODER_OUTPUT_DIM,
        Config.SPEAKER_ENCODER_EMBED_DIM,
        Config.STYLE_ENCODER_EMBED_DIM,
        Config.STYLE_MODULATOR_HIDDEN_DIM
    ).to(Config.DEVICE)
    vocoder = HiFiGANVocoder().to(Config.DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=Config.DEVICE)
    content_encoder.load_state_dict(checkpoint["content_encoder"])
    speaker_encoder.load_state_dict(checkpoint["speaker_encoder"])
    style_encoder.load_state_dict(checkpoint["style_encoder"])
    style_modulator.load_state_dict(checkpoint["style_modulator"])
    vocoder.load_state_dict(checkpoint["vocoder"])

    for model in [content_encoder, speaker_encoder, style_encoder, style_modulator, vocoder]:
        model.eval()

    input_audio = preprocess_input(INPUT_PATH).to(Config.DEVICE)
    style_audio = load_audio(STYLE_PATH)
    style_mel = extract_mel_spectrogram(style_audio, sr=Config.SAMPLE_RATE).to(Config.DEVICE)
    speaker_audio = load_audio(SPEAKER_PATH)
    speaker_mel = extract_mel_spectrogram(speaker_audio, sr=Config.SAMPLE_RATE).to(Config.DEVICE)

    with torch.no_grad():
        content = content_encoder(input_audio)  # [B, T', C]
        speaker_embed = speaker_encoder(speaker_mel)
        style_embed = style_encoder(style_mel)
        modulated = style_modulator(content, speaker_embed, style_embed)  # [B, T', C]

        # Diagnostics
        print(f"[Modulated] min: {modulated.min().item()} max: {modulated.max().item()} mean: {modulated.mean().item()}")

        # === Save numpy arrays and print detailed stats for debugging ===
        import numpy as np

        modulated_np = modulated.squeeze(0).cpu().numpy()  # [T, C]
        style_mel_np = style_mel.squeeze(0).cpu().numpy()
        speaker_mel_np = speaker_mel.squeeze(0).cpu().numpy()

        np.save("modulated_features.npy", modulated_np)
        np.save("style_mel.npy", style_mel_np)
        np.save("speaker_mel.npy", speaker_mel_np)
        print(f"✅ Saved modulated features as numpy array: modulated_features.npy")
        print(f"✅ Saved style mel spectrogram as numpy array: style_mel.npy")
        print(f"✅ Saved speaker mel spectrogram as numpy array: speaker_mel.npy")

        def print_stats(name, arr):
            print(f"[{name}] shape: {arr.shape}, min: {arr.min():.4f}, max: {arr.max():.4f}, mean: {arr.mean():.4f}, NaNs: {np.isnan(arr).any()}, infs: {np.isinf(arr).any()}")

        print_stats("Modulated features", modulated_np)
        print_stats("Style mel", style_mel_np)
        print_stats("Speaker mel", speaker_mel_np)
        # ===============================================================

        # Clamp to expected HiFi-GAN input range [-11.5, 5.5]
        modulated = torch.tanh(modulated) * 5.0  # Now roughly [-5, +5]

        # Visualizations
        visualize_modulator_output(content, modulated)
        content_mel_np = content.squeeze(0).cpu().numpy().T
        visualize_modulator_mel(content_mel_np, style_mel_np, modulated_np.T)

        # Griffin-Lim audio (diagnostic only)
        audio_griffin = griffin_lim(modulated_np.T)
        sf.write("modulated_griffin.wav", audio_griffin, Config.SAMPLE_RATE)
        print("✅ Saved Griffin-Lim audio preview of modulated mel to modulated_griffin.wav")

        # Vocoder input
        mel_like_input = modulated.transpose(1, 2)  # [B, 80, T]
        print(f"Modulated shape: {modulated.shape}")
        print(f"Vocoder input shape: {mel_like_input.shape}")

        # Vocoder input visualization
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_like_input[0].cpu().numpy(), aspect='auto', origin='lower')
        plt.title("Vocoder Input (Fixed)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("vocoder_input_fixed.png")
        plt.close()

        # Generate waveform
        output_audio = vocoder(mel_like_input).squeeze().cpu().numpy()

    # Save final audio
    sf.write(OUTPUT_PATH, output_audio, Config.SAMPLE_RATE)
    print(f"✅ Output audio saved to: {OUTPUT_PATH}")
def visualize_spectrograms():
    sr = Config.SAMPLE_RATE
    input_audio, _ = librosa.load(INPUT_PATH, sr=sr)
    style_audio, _ = librosa.load(STYLE_PATH, sr=sr)
    output_audio, _ = librosa.load(OUTPUT_PATH, sr=sr)

    def mel_db(audio):
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        return librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(12, 8))
    for i, (audio, title) in enumerate(zip(
        [input_audio, style_audio, output_audio],
        ["Input Speech", "Style Reference", "Stylized Output"]
    )):
        plt.subplot(3, 1, i + 1)
        librosa.display.specshow(mel_db(audio), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
    plt.tight_layout()
    plt.savefig(SPECTROGRAM_IMG_PATH)
    print(f"✅ Saved spectrogram comparison to {SPECTROGRAM_IMG_PATH}")
    plt.close()

if __name__ == "__main__":
    run_inference()
    visualize_spectrograms()
