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
        "robust04": _DatasetInfo(path="jxm/robust04__gtr_base__dpr", split="train"),
        "bioasq": _DatasetInfo(path="jxm/bioasq__gtr_base__dpr", split="train"),
        "trec-news": _DatasetInfo(path="jxm/trec-news__gtr_base__dpr", split="train"),
        "arguana": _DatasetInfo(path="jxm/arguana__gtr_base__dpr", split="train"),
        "quora": _DatasetInfo(path="BeIR/quora", name="corpus", split="corpus"),
        "climate-fever": _DatasetInfo(path="jxm/climate-fever__gtr_base__dpr", split="train"),
        "fever": _DatasetInfo(path="jxm/fever__gtr_base__dpr", split="train"),
        "dbpedia-entity": _DatasetInfo(path="jxm/dbpedia", split="dev"),
        "nq": _DatasetInfo(path="jxm/nq_corpus_dpr", split="dev"),
        "hotpotqa": _DatasetInfo(path="jxm/hotpotqa__gtr_base__dpr", split="train"),
        "fiqa": _DatasetInfo(path="jxm/fiqa__gtr_base__dpr", split="train"),
        "webis-touche2020": _DatasetInfo(path="jxm/webis-touche2020__gtr_base__dpr", split="train"),
        "arguana": _DatasetInfo(path="jxm/arguana__gtr_base__dpr", split="train"),
        "scidocs": _DatasetInfo(path="jxm/scidocs__gtr_base__dpr", split="train"),
        "trec-covid": _DatasetInfo(path="jxm/trec-covid__gtr_base__dpr", split="train"),
        "scifact": _DatasetInfo(path="jxm/scifact__gtr_base__dpr", split="train"),
        "nfcorpus": _DatasetInfo(path="jxm/nfcorpus__gtr_base__dpr", split="train"),
        "passwords-easy": _DatasetInfo(path="<ANON>/passwords", split="train"),
        "passwords-medium": _DatasetInfo(path="<ANON>/passwords", split="validation"),
        "passwords-hard": _DatasetInfo(path="<ANON>/passwords", split="test"),
        "msmarco": _DatasetInfo(path="jxm/msmarco__gtr_base__dpr", split="train"),
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
        full_dataset_name = f"BeIR/{path}"

        if name is None:
            return load_dataset(full_dataset_name, split=split)

        return load_dataset(full_dataset_name, name=name, split=split)["text"]
