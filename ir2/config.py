from dataclasses import dataclass

import yaml


@dataclass
class Config:
    model_name: str
    corrector_name: str
    dataset: str
    num_steps: int
    batch_size: int
    max_samples: int | None = None
    max_seq_length: int = 32
    sequence_beam_width: int = 0
    add_gaussian_noise: bool = False
    noise_lambda: float = 0.1

    @classmethod
    def load(cls, config_file: str) -> "Config":
        with open(config_file) as f:
            return cls(**yaml.load(f, Loader=yaml.FullLoader))
