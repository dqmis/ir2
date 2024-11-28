import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def calc_bleu(real_data: list[str], pred_data: list[str]) -> float:
    _real_data = [sent.split() for sent in real_data]
    _pred_data = [sent.split() for sent in pred_data]
    bleu_score = 0
    for idx, sent in enumerate(_pred_data):
        bleu_score += sentence_bleu(
            [_real_data[idx]], sent, smoothing_function=SmoothingFunction().method1
        )  # IDK if this is what they used in the paper

    return bleu_score / len(_real_data)


def calc_f1(real_data: list[str], pred_data: list[str]) -> float:
    _real_data = [set(sent.split()) for sent in real_data]
    _pred_data = [set(sent.split()) for sent in pred_data]
    f1_score = 0.0
    for idx, sent in enumerate(_real_data):
        pred_sent = _pred_data[idx]
        TP = len(
            sent.intersection(pred_sent)
        )  # get true positives by all tokens present in both sets
        FP = len(
            pred_sent - sent
        )  # get false positves as tokens in the pred sent but not in the real sent
        FN = len(
            sent - pred_sent
        )  # get false negatives as tokens in the real sent but not in the pred sent
        if TP == 0:
            f1_score += 0
            continue
        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score += (2 * prec * recall) / (prec + recall)

    return f1_score / len(real_data)


def calc_exact(real_data: list[str], pred_data: list[str]) -> float:
    correct = 0
    for idx, sent in enumerate(real_data):
        if sent == pred_data[idx]:
            correct += 1

    return correct / len(pred_data)


def eval_metrics(real_data: list[str], pred_data: list[str]) -> dict[str, float]:
    bleu = calc_bleu(real_data, pred_data)
    f1 = calc_f1(real_data, pred_data)
    exact = calc_exact(real_data, pred_data)

    return {"bleu": bleu, "f1": f1, "exact": exact}


def calc_NDCG(
    score_tensor: torch.Tensor,
    corpus_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, str]],
) -> float:
    ranking = torch.argsort(score_tensor)
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
