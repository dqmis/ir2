# How to do BEIR
import logging.config
import math
import os
import pathlib
import sys

import numpy as np
import sklearn
import sklearn.metrics
import torch
import vec2text
import wandb
from alive_progress import alive_bar
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from ir2.config import Config
from ir2.inference_model import Vec2textInferenceModel
from ir2.measures import calc_NDCG, eval_metrics
from ir2.utils import split_dataset_into_chunks


# load dataset
def load_dataset(dataset="scifact"):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset
    )
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def calc_cosine_score(query_embeddings, corpus_embeddings):
    dot = query_embeddings @ corpus_embeddings.T
    query_norm = torch.norm(query_embeddings, dim=1)
    corpus_norm = torch.norm(corpus_embeddings, dim=1)

    cosine_sim = dot / torch.outer(query_norm, corpus_norm)

    return cosine_sim


def inversion_attack_loop(config):
    infrence_model = Vec2textInferenceModel(
        model_name=config.model_name, corrector_name=config.corrector_name
    )
    corpus, queries, qrels = load_dataset(config.dataset)  # maybe make this similar to infrence?
    measures = []
    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())

    query_embeddings = infrence_model.get_embeddings(query_embeddings, add_gaussian_noise=False)
    score_tensor = torch.zeros((len(query_ids), len(corpus_ids)))
    batch_counter = 0
    # maybe add extra loop for all possible noise configs?
    lambda_list = config.noise_lambda_list
    result_dict = dict()
    for noise_lambda in lambda_list:
        for batch in tqdm(split_dataset_into_chunks(corpus_ids, config.batch_size)):
            corpus_text = [corpus[id]["text"] for id in batch]
            corpus_embeddings = infrence_model.get_embeddings(
                corpus_text,
                add_gaussian_noise=config.add_gaussian_noise,
                noise_mean=config.noise_mean,
                noise_std=config.noise_std,
                noise_lambda=noise_lambda,
            )
            cosine_sim = calc_cosine_score(query_embeddings, corpus_embeddings)
            score_tensor[
                :, batch_counter * config.batch_size : (batch_counter + 1) * config.batch_size
            ] = cosine_sim
            batch_counter += 0
            results = infrence_model.invert_embeddings(
                corpus_embeddings, num_steps=config.num_steps
            )
            measures.append(eval_metrics(corpus_text, results))
        avg_bleu = sum([m["bleu"] for m in measures]) / len(measures)
        NDCG = calc_NDCG(score_tensor, corpus_ids, query_ids, qrels)
        result_dict[noise_lambda] = {"bleu": avg_bleu, "NDCG": NDCG}
    return result_dict


if __name__ == "__main__":
    # need to figure this part out
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = Config.load(config_path)
    results = inversion_attack_loop(config)

    wandb.init(project="ir2", config=config)
    wandb.log(results)
    wandb.finish()
