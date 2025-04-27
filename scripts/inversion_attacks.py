import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import wandb
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm

from src.config import Config
from src.inference_model import Vec2textInferenceModel
from src.quantization import quantize
from src.utils import split_dataset_into_chunks
from src.vec2text_measures import compute_text_comparison_metrics


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


def calc_NDCG(
    score_tensor: torch.Tensor,
    corpus_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, str]],
) -> float:
    ranking = torch.argsort(score_tensor, descending=True)
    normelizer = np.arange(2, 12)
    normelizer = np.log2(normelizer)
    NDCG = 0.0
    for internal_query_id, query_id in enumerate(query_ids):
        relevant_docs = qrels[query_id]
        # get ideal score
        ideal_score = np.zeros(shape=(10,))  # hardcodes for NDCG @ 10
        best_scores = sorted(list(relevant_docs.values()))[::-1][:10]
        for rank, val in enumerate(best_scores):
            ideal_score[rank] = val

        # get pred score
        pred_score = np.zeros(shape=(10,))
        query_ranking = ranking[internal_query_id][:10]
        for rank, doc_id in enumerate(query_ranking):
            doc_id = corpus_ids[doc_id]
            # not all doc ids are present in relevant docs
            # give a score of zero in this case
            if doc_id in relevant_docs:
                pred_score[rank] += relevant_docs[doc_id]

        # calc NDCG
        DCG = np.sum(pred_score / normelizer)
        IDCG = np.sum(ideal_score / normelizer)
        NDCG += DCG / IDCG
    NDCG /= len(query_ids)
    return NDCG


def inversion_attack_loop(config):
    torch.cuda.empty_cache()
    inference_model = Vec2textInferenceModel(
        model_name=config.model_name, corrector_name=config.corrector_name
    )
    if not config.quantize:
        method_list = config.noise_lambda
    else:
        method_list = config.quantize_list

    result_dict = dict()
    dataset_list = config.dataset_list
    for dataset in dataset_list:
        print(dataset)
        # load corpus
        if dataset in result_dict:
            continue
        corpus, queries, qrels = load_dataset(dataset)
        query_ids = list(queries.keys())[: config.max_querry_samples]
        corpus_ids = list(corpus.keys())
        query_text = [queries[id] for id in query_ids]
        # get query embeddings
        query_embeddings, query_token_ids = inference_model.get_embeddings(
            query_text, add_gaussian_noise=False
        )

        # set up scores matrix
        score_tensor = torch.zeros((len(method_list), len(query_ids), len(corpus_ids)))
        batch_counter = 0
        target_strings = []
        pred_strings = [[] for _ in range(len(method_list))]
        # attack loop
        for batch in tqdm(split_dataset_into_chunks(corpus_ids, config.batch_size)):
            # embed the data first and then apply data permutation
            # to save on embedding time
            corpus_text = [corpus[id]["text"] for id in batch]
            corpus_embeddings, corpus_token_ids = inference_model.get_embeddings(
                corpus_text, add_gaussian_noise=False, max_length=config.max_seq_length
            )
            if batch_counter < config.max_samples:
                target_strings.extend(inference_model.batch_decode(corpus_token_ids))
            for method_idx, method_val in enumerate(method_list):
                # add noise
                if not config.quantize:
                    noise = method_val * torch.normal(mean=0, std=1, size=corpus_embeddings.size())
                    if torch.cuda.is_available():
                        noise = noise.to("cuda")
                    permutated_embeddings = corpus_embeddings.detach().clone()
                    permutated_embeddings += noise
                # quantization
                else:
                    permutated_embeddings = quantize(
                        corpus_embeddings,
                        method=method_val,
                        max_val=config.quant_max_val,
                        min_val=config.quant_min_val,
                    )

                # retrieval
                cosine_sim = calc_cosine_score(query_embeddings, permutated_embeddings)
                score_tensor[
                    method_idx,
                    :,
                    batch_counter * config.batch_size : batch_counter * config.batch_size + len(batch),
                ] = cosine_sim

                # reconstruction
                # only do reconstruction for max amount of samples
                if batch_counter < config.max_samples:
                    pred_text = inference_model.invert_embeddings(
                        permutated_embeddings,
                        num_steps=config.num_steps,
                        max_length=config.max_seq_length,
                    )

                    pred_strings[method_idx].extend(pred_text)

            batch_counter += 1

        # save results
        result_dict[dataset] = dict()
        target_ids = inference_model.batch_encode_plus(target_strings)["input_ids"]
        for method_idx, method_val in enumerate(method_list):
            # calc reconstruction metrics
            method_pred_strings = pred_strings[method_idx]
            method_pred_ids = inference_model.batch_encode_plus(method_pred_strings)["input_ids"]
            metrics = compute_text_comparison_metrics(
                predictions_ids=method_pred_ids.tolist(),
                predictions_str=method_pred_strings,
                references_ids=target_ids.tolist(),
                references_str=target_strings,
            )
            bleu_score = metrics["bleu_score"]
            bleu_score_sem = metrics["bleu_score_sem"]

            # calc NDCG score
            NDCG = calc_NDCG(score_tensor[method_idx], corpus_ids, query_ids, qrels)

            # save results
            if not config.quantize:
                key = f"lambda noise {method_val}"
            else:
                key = f"method: {method_val}"

            result_dict[dataset][key] = {
                "bleu": bleu_score,
                "bleu_score_sem": bleu_score_sem,
                "ndcg": NDCG,
            }

        # export results
        with open(config.export_path, "wb") as f:
            pickle.dump(result_dict, f)

        torch.cuda.empty_cache()
    return result_dict


if __name__ == "__main__":
    print("GPU available?", torch.cuda.is_available())
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = Config.load(config_path)
    results = inversion_attack_loop(config)
    wandb.init(project="vec2text-repro", config=config)
    wandb.log(results)
    wandb.finish()
