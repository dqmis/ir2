import collections

import evaluate
import nltk
import numpy as np
import scipy
import vec2text


def mean(L: list[int] | list[float]) -> float:
    return sum(L) / len(L)


def sem(L: list[float]) -> float:
    result: float = scipy.stats.sem(np.array(L))
    if isinstance(result, np.ndarray):
        result_: float = result.mean().item()
        return result_
    return result


def _count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
    ngrams_1 = nltk.ngrams(s1, n)
    ngrams_2 = nltk.ngrams(s2, n)
    ngram_counts_1 = collections.Counter(ngrams_1)
    ngram_counts_2 = collections.Counter(ngrams_2)
    total = 0
    for ngram, count in ngram_counts_1.items():
        total += min(count, ngram_counts_2[ngram])
    return total


def compute_text_comparison_metrics(
    predictions_ids: list[list[int]],
    predictions_str: list[str],
    references_ids: list[list[int]],
    references_str: list[str],
) -> dict[str, float]:
    nltk.download("punkt")
    nltk.download("punkt_tab")

    assert len(predictions_ids) == len(references_ids)
    assert len(predictions_ids) == len(predictions_str)
    assert len(predictions_str) == len(references_str)
    num_preds = len(predictions_ids)
    if not num_preds:
        return {}

    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")

    precision_sum = 0.0
    recall_sum = 0.0
    num_overlapping_words = []
    num_overlapping_bigrams = []
    num_overlapping_trigrams = []
    num_true_words = []
    num_pred_words = []
    f1s = []

    for i in range(num_preds):
        true_words = nltk.tokenize.word_tokenize(references_str[i])
        pred_words = nltk.tokenize.word_tokenize(predictions_str[i])
        num_true_words.append(len(true_words))
        num_pred_words.append(len(pred_words))

        true_words_set = set(true_words)
        pred_words_set = set(pred_words)
        TP = len(true_words_set & pred_words_set)
        FP = len(true_words_set) - len(true_words_set & pred_words_set)
        FN = len(pred_words_set) - len(true_words_set & pred_words_set)

        precision = (TP) / (TP + FP + 1e-20)
        recall = (TP) / (TP + FN + 1e-20)

        try:
            f1 = (2 * precision * recall) / (precision + recall + 1e-20)
        except ZeroDivisionError:
            f1 = 0.0
        f1s.append(f1)

        precision_sum += precision
        recall_sum += recall

        num_overlapping_words.append(_count_overlapping_ngrams(true_words, pred_words, 1))
        num_overlapping_bigrams.append(_count_overlapping_ngrams(true_words, pred_words, 2))
        num_overlapping_trigrams.append(_count_overlapping_ngrams(true_words, pred_words, 3))

    set_token_metrics = {
        "token_set_precision": (precision_sum / num_preds),
        "token_set_recall": (recall_sum / num_preds),
        "token_set_f1": mean(f1s),
        "token_set_f1_sem": sem(f1s),
        "n_ngrams_match_1": mean(num_overlapping_words),
        "n_ngrams_match_2": mean(num_overlapping_bigrams),
        "n_ngrams_match_3": mean(num_overlapping_trigrams),
        "num_true_words": mean(num_true_words),
        "num_pred_words": mean(num_pred_words),
    }

    bleu_results = np.array(
        [
            bleu_metric.compute(predictions=[p], references=[r])["score"]
            for p, r in zip(predictions_str, references_str)
        ]
    ).tolist()

    rouge_result = rouge_metric.compute(predictions=predictions_str, references=references_str)
    exact_matches = np.array(predictions_str) == np.array(references_str)
    gen_metrics = {
        "bleu_score": mean(bleu_results),
        "bleu_score_sem": sem(bleu_results),
        "rouge_score": rouge_result["rouge1"],
        "exact_match": mean(exact_matches),
        "exact_match_sem": sem(exact_matches),
    }

    all_metrics = {**set_token_metrics, **gen_metrics}

    for metric in [vec2text.metrics.EmbeddingCosineSimilarity()]:
        all_metrics.update(metric(references_str, predictions_str))

    return all_metrics
