"""Audio and latent clocks shared by the independent codec and its consumers."""

AUDIO_SAMPLE_RATE = 48000
LATENT_HOP_LENGTH = 960
LATENT_FRAMES_PER_SECOND = AUDIO_SAMPLE_RATE // LATENT_HOP_LENGTH
DEFAULT_LATENT_DIM = 128
