import logging.config
import random
import sys

import torch
from tqdm import tqdm
from vec2text.trainers.base import BaseTrainer

import wandb
from ir2.config import Config
from ir2.dataset_loader import DatasetLoader
from ir2.inference_model import Vec2textInferenceModel
from ir2.utils import split_dataset_into_chunks

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_name_recovery_rates() -> dict[str, float]:
    # TODO: For table 3, rates of recovery for first, last, and full names
    pass


def inference_loop(config: Config):
    inference_model = Vec2textInferenceModel(
        model_name=config.model_name, corrector_name=config.corrector_name
    )

    ds = DatasetLoader.load(config.dataset)
    ds = random.sample(ds, config.max_samples)

    all_predictions = []
    all_references = []
    all_pred_embeddings = []
    all_ref_embeddings = []

    tokenizer = inference_model._tokenizer

    for batch in tqdm(split_dataset_into_chunks(ds, config.batch_size)):
        embeddings = inference_model.get_embeddings(batch).to(device)
        results = inference_model.invert_embeddings(embeddings, num_steps=config.num_steps)

        all_references.extend(batch)
        all_predictions.extend(results)

        pred_embeddings = inference_model.get_embeddings(results).to(device)
        ref_embeddings = embeddings

        all_pred_embeddings.append(pred_embeddings)
        all_ref_embeddings.append(ref_embeddings)

    all_pred_embeddings = torch.cat(all_pred_embeddings, dim=0)
    all_ref_embeddings = torch.cat(all_ref_embeddings, dim=0)

    decoded_preds = all_predictions
    decoded_labels = all_references

    predictions_ids = tokenizer.batch_encode_plus(
        decoded_preds, return_tensors="pt", padding=True
    )["input_ids"].to(device)
    references_ids = tokenizer.batch_encode_plus(
        decoded_labels, return_tensors="pt", padding=True
    )["input_ids"].to(device)

    trainer = BaseTrainer(
        model=inference_model._encoder.to(device), tokenizer=inference_model._tokenizer
    )
    trainer.enable_emb_cos_sim_metric()

    # TODO: Check if we should use the additional logic from trainer.evaluation_loop
    metrics = trainer._text_comparison_metrics(
        predictions_ids=predictions_ids.tolist(),
        predictions_str=decoded_preds,
        references_ids=references_ids.tolist(),
        references_str=decoded_labels,
    )

    if config.dataset == "mimic-iii":  # Relevant for table 3
        name_recovery_metrics = compute_name_recovery_rates(decoded_preds, decoded_labels)
        metrics.update(name_recovery_metrics)

    return metrics


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = Config.load(config_path)
    results = inference_loop(config)

    wandb.init(project="ir2", config=config)
    wandb.log(results)
    wandb.finish()
