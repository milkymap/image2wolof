import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VariablesConfig(BaseSettings):
    """Every value can come from .env, the environment, or the `image2wolof` options (which take precedence)."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    # --- frozen vision backbone ---
    siglip_model: str = Field(validation_alias="SIGLIP_MODEL", default="google/siglip2-base-patch16-224")
    siglip_revision: str = Field(validation_alias="SIGLIP_REVISION", default="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2")
    device: str = Field(validation_alias="DEVICE", default="auto")  # auto | cuda | cpu

    # --- tokenizer ---
    vocab_size: int = Field(validation_alias="VOCAB_SIZE", default=4000)

    # --- decoder ---
    layers: int = Field(validation_alias="LAYERS", default=6)
    d_model: int = Field(validation_alias="D_MODEL", default=512)
    heads: int = Field(validation_alias="HEADS", default=8)
    ff: int = Field(validation_alias="FF", default=2048)
    dropout: float = Field(validation_alias="DROPOUT", default=0.1)

    # --- training ---
    epochs: int = Field(validation_alias="EPOCHS", default=20)
    batch_size: int = Field(validation_alias="BATCH_SIZE", default=128)
    lr: float = Field(validation_alias="LR", default=3e-4)
    weight_decay: float = Field(validation_alias="WEIGHT_DECAY", default=0.01)
    warmup: int = Field(validation_alias="WARMUP", default=200)
    label_smoothing: float = Field(validation_alias="LABEL_SMOOTHING", default=0.1)
    clip: float = Field(validation_alias="CLIP", default=1.0)
    patience: int = Field(validation_alias="PATIENCE", default=4)
    seed: int = Field(validation_alias="SEED", default=42)
    log_every: int = Field(validation_alias="LOG_EVERY", default=100)

    # --- decoding ---
    max_tokens: int = Field(validation_alias="MAX_TOKENS", default=96)

    # --- server ---
    host: str = Field(validation_alias="HOST", default="127.0.0.1")
    port: int = Field(validation_alias="PORT", default=8080)
    max_files: int = Field(validation_alias="MAX_FILES", default=32)
    max_mb: int = Field(validation_alias="MAX_MB", default=25)

    def torch_device(self) -> str:
        """Resolve `auto` to cuda when available."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("DEVICE=cuda requested but CUDA is unavailable")
        return self.device
