from dataclasses import dataclass

import yaml


@dataclass
class Config:
    model_name: str
    corrector_name: str
    dataset: str
    num_steps: int
    batch_size: int
    add_gaussian_noise: bool = False
    noise_mean: float = 0
    noise_std: float = 0.1
    noise_lambda: float = 0.1
    max_samples: int

    @classmethod
    def load(cls, config_file: str) -> "Config":
        with open(config_file) as f:
            return cls(**yaml.load(f, Loader=yaml.FullLoader))
