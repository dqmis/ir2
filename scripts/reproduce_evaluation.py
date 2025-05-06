import logging.config
import random
import sys

import wandb
from tqdm import tqdm

from vec2text-repro.config import Config
from vec2text-repro.dataset_loader import DatasetLoader
from vec2text-repro.inference_model import Vec2textInferenceModel
from vec2text-repro.utils import split_dataset_into_chunks
from vec2text-repro.vec2text_measures import compute_text_comparison_metrics

random.seed(42)

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)


def compute_name_recovery_rates() -> dict[str, float]:
    # TODO: For table 3, rates of recovery for first, last, and full names
    {"foo": 0.0, "bar": 0.0}


def inference_loop(config: Config):
    inference_model = Vec2textInferenceModel(
        model_name=config.model_name, corrector_name=config.corrector_name
    )

    print("Loading data...")
    ds = DatasetLoader.load(config.dataset)
    ds = random.sample(ds, config.max_samples) if config.max_samples else ds

    prediction_strs = []
    reference_strs = []

    print("Running inference...")

    for batch in tqdm(split_dataset_into_chunks(ds, config.batch_size)):
        input_embeddings, input_tokens = inference_model.get_embeddings(
            batch,
            max_length=config.max_seq_length,
            add_gaussian_noise=config.add_gaussian_noise,
            noise_lambda=config.noise_lambda,
        )

        prediction_str = inference_model.invert_embeddings(
            input_embeddings,
            num_steps=config.num_steps,
            max_length=config.max_seq_length,
            sequence_beam_width=config.sequence_beam_width,
            do_sample=config.do_sample,
            top_p=config.top_p,
        )

        prediction_strs.extend(prediction_str)
        reference_strs.extend(inference_model.batch_decode(input_tokens))

    predictions_ids = inference_model.batch_encode_plus(prediction_strs)["input_ids"]
    references_ids = inference_model.batch_encode_plus(reference_strs)["input_ids"]

    print("Computing metrics...")
    metrics = compute_text_comparison_metrics(
        predictions_ids=predictions_ids.tolist(),
        predictions_str=prediction_strs,
        references_ids=references_ids.tolist(),
        references_str=reference_strs,
    )

    if config.dataset == "mimic-iii":  # Relevant for table 3
        name_recovery_metrics = compute_name_recovery_rates(prediction_strs, reference_strs)
        metrics.update(name_recovery_metrics)

    return metrics


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = Config.load(config_path)

    wandb.init(project="vec2text-repro", config=config)
    wandb.run.name = f"model-{config.model_name}_corrector-{config.corrector_name}_steps-{config.num_steps}_beam-{config.sequence_beam_width}_nucleus-{config.do_sample}"

    results = inference_loop(config)

    wandb.log(results)
    wandb.finish()
