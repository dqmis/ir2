from dataclasses import dataclass
from typing import Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

DatasetType = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]


@dataclass
class _DatasetInfo:
    path: str
    name: str | None = None
    split: str = "train"


class DatasetLoader:
    _DATASET_MAP = {
        "signal1m": _DatasetInfo(path="BeIR/signal1m-generated-queries"),
        "robust04": _DatasetInfo(path="BeIR/robust04-generated-queries"),
        "bioasq": _DatasetInfo(path="BeIR/bioasq-generated-queries"),
        "trec-news": _DatasetInfo(path="BeIR/trec-news-generated-queries"),
        "quora": _DatasetInfo(path="BeIR/quora", name="corpus", split="corpus"),
        "climate-fever": _DatasetInfo(path="BeIR/climate-fever", name="corpus", split="corpus"),
        "fever": _DatasetInfo(path="BeIR/fever", name="corpus", split="corpus"),
        "dbpedia-entity": _DatasetInfo(path="BeIR/dbpedia-entity", name="corpus", split="corpus"),
        "nq": _DatasetInfo(path="jxm/nq_corpus_dpr", split="dev"),
        "hotpotqa": _DatasetInfo(path="BeIR/hotpotqa", name="corpus", split="corpus"),
        "fiqa": _DatasetInfo(path="BeIR/fiqa", name="corpus", split="corpus"),
        "webis-touche2020": _DatasetInfo(path="BeIR/webis-touche2020", name="corpus", split="corpus"),
        "arguana": _DatasetInfo(path="BeIR/arguana", name="corpus", split="corpus"),
        "scidocs": _DatasetInfo(path="BeIR/scidocs", name="corpus", split="corpus"),
        "trec-covid": _DatasetInfo(path="BeIR/trec-covid", name="corpus", split="corpus"),
        "scifact": _DatasetInfo(path="BeIR/scifact", name="corpus", split="corpus"),
        "nfcorpus": _DatasetInfo(path="BeIR/nfcorpus", name="corpus", split="corpus"),
        "msmarco": _DatasetInfo(path="Tevatron/msmarco-passage-corpus", name="corpus", split="corpus"),
    }

    @classmethod
    def load(cls, dataset_name: str) -> DatasetType:
        dataset_info = cls._DATASET_MAP.get(dataset_name)
        if dataset_info is None:
            raise ValueError(
                f"dataset {dataset_name} is not a suported dataset pick a dataset from \n {cls._DATASET_MAP.keys()}"  # noqa
            )

        return cls._load_dataset(dataset_info.path, dataset_info.split, dataset_info.name)

    @classmethod
    def _load_dataset(cls, path: str, split: str, name: str | None = None) -> DatasetType:

        if name is None:
            return load_dataset(path, split=split)["text"]

        return load_dataset(path, name=name, split=split)["text"]
