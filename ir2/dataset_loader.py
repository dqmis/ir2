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
        "signal1m": _DatasetInfo(path="signal1m-generated-queries"),
        "robust04": _DatasetInfo(path="robust04-generated-queries"),
        "bioasq": _DatasetInfo(path="bioasq-generated-queries"),
        "trec-news": _DatasetInfo(path="trec-news-generated-queries"),
        "quora": _DatasetInfo(path="quora", name="corpus", split="corpus"),
        "climate-fever": _DatasetInfo(path="climate-fever", name="corpus", split="corpus"),
        "fever": _DatasetInfo(path="fever", name="corpus", split="corpus"),
        "dbpedia-entity": _DatasetInfo(path="dbpedia-entity", name="corpus", split="corpus"),
        "nq": _DatasetInfo(path="nq", name="corpus", split="corpus"),
        "hotpotqa": _DatasetInfo(path="hotpotqa", name="corpus", split="corpus"),
        "fiqa": _DatasetInfo(path="fiqa", name="corpus", split="corpus"),
        "webis-touche2020": _DatasetInfo(path="webis-touche2020", name="corpus", split="corpus"),
        "arguana": _DatasetInfo(path="arguana", name="corpus", split="corpus"),
        "scidocs": _DatasetInfo(path="scidocs", name="corpus", split="corpus"),
        "trec-covid": _DatasetInfo(path="trec-covid", name="corpus", split="corpus"),
        "scifact": _DatasetInfo(path="scifact", name="corpus", split="corpus"),
        "nfcorpus": _DatasetInfo(path="nfcorpus", name="corpus", split="corpus"),
        "msmarco": _DatasetInfo(path="msmarco", name="corpus", split="corpus"),
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

        return load_dataset(full_dataset_name, name=name, split=split)
