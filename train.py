import os
import time

import kagglehub
import torch
import torch.optim as optim
import torch.cuda.amp as amp

from config import Config
from my_models.content_encoder import ContentEncoder
from my_models.speaker_encoder import SpeakerEncoder
from my_models.style_encoder import StyleEncoder
from my_models.vocoder import HiFiGANVocoder
from my_modules.style_modulator import StyleModulator
from my_modules.transfer_module import train_transfer_step
from my_utils.data import RAVDESSDataset, create_dataloader


def main():
    # 1. Data Preparation
    print("Downloading RAVDESS dataset...")
    DATA_PATH = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    print("Dataset downloaded at:", DATA_PATH)

    dataset = RAVDESSDataset(
        DATA_PATH, Config.AUDIO_LEN * Config.SAMPLE_RATE, Config.SAMPLE_RATE
    )
    dataloader = create_dataloader(dataset, Config.BATCH_SIZE, shuffle=True)

    # 2. Model Initialization
    content_encoder = ContentEncoder(
        output_dim=Config.CONTENT_ENCODER_OUTPUT_DIM,
        pretrained=True,
        freeze_base=True  # Freeze base Wav2Vec2 to speed up training
    ).to(Config.DEVICE)

    speaker_encoder = SpeakerEncoder(
        Config.N_MELS, Config.SPEAKER_ENCODER_EMBED_DIM
    ).to(Config.DEVICE)

    style_encoder = StyleEncoder(
        Config.N_MELS, Config.STYLE_ENCODER_EMBED_DIM
    ).to(Config.DEVICE)

    style_modulator = StyleModulator(
        Config.CONTENT_ENCODER_OUTPUT_DIM,
        Config.SPEAKER_ENCODER_EMBED_DIM,
        Config.STYLE_ENCODER_EMBED_DIM,
        Config.STYLE_MODULATOR_HIDDEN_DIM,
    ).to(Config.DEVICE)

    vocoder = HiFiGANVocoder().to(Config.DEVICE)

    # Freeze encoders to prevent training
    for param in content_encoder.parameters():
        param.requires_grad = False
    for param in speaker_encoder.parameters():
        param.requires_grad = False
    for param in style_encoder.parameters():
        param.requires_grad = False

    # Set frozen encoders to eval mode
    content_encoder.eval()
    speaker_encoder.eval()
    style_encoder.eval()

    # Only train style_modulator and vocoder
    optimizer = optim.Adam(
        list(style_modulator.parameters()) + list(vocoder.parameters()),
        lr=Config.LEARNING_RATE,
    )

    scaler = amp.GradScaler()

    # 4. Training Loop
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nStarting Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        start_epoch_time = time.time()
        total_loss = 0
        num_batches = len(dataloader)

        for i, (wave, mel) in enumerate(dataloader):
            batch_start_time = time.time()

            wave = wave.to(Config.DEVICE)
            mel = mel.to(Config.DEVICE)

            optimizer.zero_grad()
            with amp.autocast():
                losses = train_transfer_step(
                    content_encoder,
                    speaker_encoder,
                    style_encoder,
                    style_modulator,
                    vocoder,
                    wave,
                    mel,
                    Config,
                )

            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses["loss"]

            if i % 10 == 0 or i == num_batches - 1:
                elapsed = time.time() - batch_start_time
                avg_loss = total_loss / (i + 1)
                print(
                    f"Epoch [{epoch + 1}/{Config.NUM_EPOCHS}] "
                    f"Batch [{i + 1}/{num_batches}] "
                    f"Loss: {losses['loss']:.4f} "
                    f"Avg Loss: {avg_loss:.4f} "
                    f"Batch time: {elapsed:.2f}s"
                )

        epoch_time = time.time() - start_epoch_time
        avg_epoch_loss = total_loss / num_batches
        print(
            f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds. "
            f"Average Loss: {avg_epoch_loss:.4f}"
        )

    # 5. Save Model
    torch.save(
        {
            "content_encoder": content_encoder.state_dict(),
            "speaker_encoder": speaker_encoder.state_dict(),
            "style_encoder": style_encoder.state_dict(),
            "style_modulator": style_modulator.state_dict(),
            "vocoder": vocoder.state_dict(),
        },
        "speech_style_transfer_model.pth",
    )


if __name__ == "__main__":
    main()
