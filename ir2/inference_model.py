import torch
import vec2text
from transformers import AutoModel, AutoTokenizer


class Vec2textInferenceModel:
    def __init__(self, model_name: str, corrector_name: str):
        self._encoder = AutoModel.from_pretrained(model_name).encoder
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._corrector = vec2text.load_pretrained_corrector(corrector_name)

        if self._cuda_is_available():
            self._encoder = self._encoder.to("cuda")

    def _cuda_is_available(self) -> bool:
        return torch.cuda.is_available()

    def get_embeddings(
        self,
        text_list: list[str],
        max_length: int = 128,
        truncation: bool = True,
        padding: str = "max_length",
        add_gaussian_noise: bool = False,
        noise_mean: float = 0,
        noise_std: float = 0.1,
        noise_lambda: float = 0.1,
    ) -> torch.Tensor:
        inputs = self._tokenizer(
            text_list,
            return_tensors="pt",
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )

        if self._cuda_is_available():
            inputs = inputs.to("cuda")

        with torch.no_grad():
            model_output = self._encoder(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            hidden_state = model_output.last_hidden_state
            embeddings: torch.Tensor = vec2text.models.model_utils.mean_pool(
                hidden_state, inputs["attention_mask"]
            )

            if add_gaussian_noise:
                noise = noise_lambda * torch.normal(mean=0, std=1, size=embeddings.size())
                if self._cuda_is_available():
                    noise = noise.to("cuda")
                embeddings += noise

        return embeddings

    def invert_embeddings(
        self,
        embeddings: torch.Tensor,
        num_steps: int,
    ) -> list[str]:
        if self._cuda_is_available():
            embeddings = embeddings.to("cuda")
        else:
            embeddings = embeddings.to(self._corrector.model.device)

        inverted_embeddings: list[str] = vec2text.invert_embeddings(
            embeddings=embeddings,
            corrector=self._corrector,
            num_steps=num_steps,
        )

        return inverted_embeddings
