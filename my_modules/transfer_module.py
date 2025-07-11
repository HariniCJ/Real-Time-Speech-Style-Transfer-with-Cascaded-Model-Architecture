def train_transfer_step(
    content_encoder,
    speaker_encoder,
    style_encoder,
    style_modulator,
    vocoder,
    wave,
    mel,
    Config,
):
    # Removed: content_encoder.train()
    # Removed: speaker_encoder.train()
    # Removed: style_encoder.train()

    style_modulator.train()
    vocoder.train()

    # Content features
    content_features = content_encoder(wave)

    # Mel preprocessing
    mel = mel.squeeze(1)
    mel = torch.nan_to_num(mel, nan=0.0, posinf=1.0, neginf=-1.0)
    mel = torch.clamp(mel, min=0.0, max=10.0)

    speaker_embedding = speaker_encoder(mel)
    style_embedding = style_encoder(mel)

    # Style modulation
    modulated_content = style_modulator(content_features, speaker_embedding, style_embedding)

    modulated_content = modulated_content.permute(0, 2, 1)

    reconstructed_audio = vocoder(modulated_content)

    rec_audio = reconstructed_audio.squeeze(1)
    target_wave = wave.squeeze(1)
    min_len = min(rec_audio.size(1), target_wave.size(1))
    rec_audio = rec_audio[:, :min_len]
    target_wave = target_wave[:, :min_len]

    loss = nn.functional.l1_loss(rec_audio, target_wave)

    return {"loss": loss.item(), "reconstruction_loss": loss.item()}
