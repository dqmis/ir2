import logging.config
import sys

<<<<<<< HEAD
from tqdm import tqdm

import wandb
=======
import wandb
from tqdm import tqdm

>>>>>>> ed39763 (Removing uva connection)
from ir2.config import Config
from ir2.dataset_loader import DatasetLoader
from ir2.inference_model import Vec2textInferenceModel
from ir2.measures import eval_metrics
from ir2.utils import split_dataset_into_chunks

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)


def infrence_loop(config: Config):
    infrence_model = Vec2textInferenceModel(
        model_name=config.model_name, corrector_name=config.corrector_name
    )
    ds = DatasetLoader.load(config.dataset)

    measures = []

    for batch in tqdm(split_dataset_into_chunks(ds, config.batch_size)):
        embeddings = infrence_model.get_embeddings(
            batch,
            add_gaussian_noise=config.add_gaussian_noise,
            noise_mean=config.noise_mean,
            noise_std=config.noise_std,
        )
        results = infrence_model.invert_embeddings(
            embeddings, num_steps=config.num_steps
        )

        measures.append(eval_metrics(batch, results))

    avg_bleu = sum([m["bleu"] for m in measures]) / len(measures)
    avg_f1 = sum([m["f1"] for m in measures]) / len(measures)
    avg_exact = sum([m["exact"] for m in measures]) / len(measures)

    return {"bleu": avg_bleu, "f1": avg_f1, "exact": avg_exact}


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = Config.load(config_path)
    results = infrence_loop(config)

    wandb.init(project="vec2text-repro", config=config)
    wandb.log(results)
    wandb.finish()
