from datasets import Dataset


def split_dataset_into_chunks(dataset: Dataset, chunk_size: int) -> list[Dataset]:
    chunks = []
    for i in range(0, len(dataset), chunk_size):
        chunks.append(dataset[i : i + chunk_size])  # noqa
    return chunks
