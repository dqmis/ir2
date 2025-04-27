from dataclasses import dataclass, field

import yaml


@dataclass
class Config:
    model_name: str
    corrector_name: str
    dataset: str
    num_steps: int
    batch_size: int
    max_samples: int | None = None
    add_gaussian_noise: bool = False
    noise_mean: float = 0
    noise_std: float = 0.1
    noise_lambda: list = field(default_factory=list)
    dataset_list: list = field(default_factory=list)
    quantize_list: list = field(default_factory=list)
    max_querry_samples: int = 100
    max_seq_length: int = 32
    sequence_beam_width: int = 0
    do_sample: bool = False
    top_p: float | None = None
    export_path: str = "out/results.pickle"
    quantize: bool = False
    quant_max_val = 1.5
    quant_min_val = -1.5

    @classmethod
    def load(cls, config_file: str) -> "Config":
        with open(config_file) as f:
            return cls(**yaml.load(f, Loader=yaml.FullLoader))
