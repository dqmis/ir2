from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
import torch

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

def calc_NDCG(score_tensor, corpus_ids, query_ids, qrels):
    ranking = torch.argsort(score_tensor)
    normelizer = np.arange(2, 12)
    normelizer = np.log2(normelizer)
    NDCG = 0
    for query_id in query_ids:
        print(qrels[query_ids[0]]) 
        relevant_docs = qrels[query_id]
        ideal_score = np.array(sorted(list(relevant_docs.values()))[::-1][:10])
        query_ranking = ranking[query_id][:10]
        pred_score = np.array([relevant_docs[corpus_ids[doc_id]] for doc_id in query_ranking])
        DCG = np.sum(pred_score / normelizer)
        IDCG = np.sum(ideal_score / normelizer)
        NDCG += DCG / IDCG
    NDCG /= len(query_ids)
    return NDCG