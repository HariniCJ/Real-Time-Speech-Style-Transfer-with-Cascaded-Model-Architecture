import torch


class Config:

    # --- Data ---
    SAMPLE_RATE = 16000
    AUDIO_LEN = 3  # Seconds
    DATA_DIR = "data/RAVDESS"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Model ---
    N_MELS = 80  # Number of Mel bands
    CONTENT_ENCODER_OUTPUT_DIM = 256
    SPEAKER_ENCODER_EMBED_DIM = 128
    STYLE_ENCODER_EMBED_DIM = 128
    STYLE_MODULATOR_HIDDEN_DIM = 512

    # --- Training ---
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    SPEAKER_ENCODER_TRIPLET_MARGIN = 0.2

    # --- Vocoder ---
    VOCODER_TYPE = "HiFiGAN"  # or "MelGAN"
