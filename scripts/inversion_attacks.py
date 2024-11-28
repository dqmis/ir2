import os
import pathlib
import pickle
import sys

import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm

import wandb
from ir2.config import Config
from ir2.inference_model import Vec2textInferenceModel
from ir2.measures import calc_NDCG, eval_metrics
from ir2.utils import split_dataset_into_chunks


# load dataset
def load_dataset(dataset="scifact"):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    # Provide the data_path where scifact has been downloaded and unzipped
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

    query_ids = list(queries.keys())
    corpus_ids = list(corpus.keys())
    query_text = [queries[id] for id in query_ids]
    query_embeddings = infrence_model.get_embeddings(query_text, add_gaussian_noise=False)

    lambda_list = config.noise_lambda
    result_dict = dict()
    # sample only for reconstruction
    # maybe first loop over batches and then over noises
    score_tensor = torch.zeros((len(lambda_list), len(query_ids), len(corpus_ids)))
    measures = [[] for _ in range(len(lambda_list))]
    batch_counter = 0
    for batch in tqdm(split_dataset_into_chunks(corpus_ids, config.batch_size)):
        # embed the data first and then add noise
        # to save on embedding time
        corpus_text = [corpus[id]["text"] for id in batch]
        corpus_embeddings = infrence_model.get_embeddings(
            corpus_text,
            add_gaussian_noise=False
        )
        for lambda_idx, noise_lambda in enumerate(lambda_list):
            # add noise
            noise = noise_lambda * torch.normal(mean=0, std=1, size=corpus_embeddings.size())
            if torch.cuda.is_available():
                noise = noise.to("cuda")
            noisy_embeddings = corpus_embeddings.detach().clone()
            noisy_embeddings += noise
            # retrieval
            cosine_sim = calc_cosine_score(query_embeddings, noisy_embeddings)
            score_tensor[lambda_idx, 
                :,
                batch_counter * config.batch_size : batch_counter * config.batch_size + len(batch),
            ] = cosine_sim
            

            # reconstruction
            # only do reconstruction for max amount of samples
            if batch_counter < config.max_samples:
                results = infrence_model.invert_embeddings(
                    noisy_embeddings, num_steps=config.num_steps
                )
                measures[lambda_idx].append(eval_metrics(corpus_text, results))
        
        batch_counter += 1
    # save results
    for lambda_idx, noise_lambda in enumerate(lambda_list): 
        avg_bleu = sum([m["bleu"] for m in measures[lambda_idx]]) / len(measures[lambda_idx])
        NDCG = calc_NDCG(score_tensor[lambda_idx], corpus_ids, query_ids, qrels)
        result_dict[f"lambda noise {noise_lambda}"] = {"bleu": avg_bleu, "ndcg": NDCG}
    return result_dict


if __name__ == "__main__":
    # need to figure this part out
    print("GPU available?", torch.cuda.is_available())
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = Config.load(config_path)
    results = inversion_attack_loop(config)
    wandb.init(project="ir2", config=config)
    wandb.log(results)
    wandb.finish()
    with open("out/scifact.pickle", "wb") as f:
        pickle.dump(results, f)
