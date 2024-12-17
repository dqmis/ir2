import torch


def naive_8_bit(embeddings: torch.Tensor, max_val: float) -> torch.Tensor:
    X_q = torch.round(127/max_val * embeddings)
    X_dq = max_val / 127 * X_q
    return X_dq


def zero_point(embeddings: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    scale = 255 / (max_val - min_val)
    zero_point = -round(scale * min_val) - 128
    X_q = torch.round(scale * embeddings + zero_point)
    X_dq = (X_q - zero_point) / scale
    return X_dq


def quantize(embeddings: torch.Tensor,
             min_val: float,
             max_val: float,
             method: str = "naive-8-bit"
             ) -> torch.Tensor:
    if method == "naive-8-bit":
        return naive_8_bit(embeddings, max_val)
    elif method == "zero_point":
        return zero_point(embeddings, min_val, max_val)
    else:
        return embeddings
