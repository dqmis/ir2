import copy

import torch
import vec2text
from transformers import AutoModel, AutoTokenizer
import openai
from transformers import GPT2TokenizerFast



class OpenAIEncoder:
    def __init__(self) -> None:
        self._client = openai.OpenAI()
        self._model = "text-embedding-ada-002"

    def __call__(
        self,
        text_list: list[str],
    ) -> torch.Tensor:
        response = self._client.embeddings.create(
            input=text_list,
            model=self._model,
            encoding_format="float",
        )
        return torch.tensor(([e.embedding for e in response.data]))



class Vec2textInferenceModel:

    def __init__(self, model_name: str, corrector_name: str):

        self._corrector = vec2text.load_pretrained_corrector(corrector_name)


        if "ada" in model_name:
            self._encoder = OpenAIEncoder()
            self._tokenizer = self._corrector.tokenizer
        else:
            self._encoder = AutoModel.from_pretrained(model_name).encoder
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self._cuda_is_available():
                self._encoder = self._encoder.to("cuda")


    def _cuda_is_available(self) -> bool:
        return torch.cuda.is_available()

    def _get_non_openai_embeddings(
        self,
        text_list: list[str],
        max_length: int = 128,
        truncation: bool = True,
        padding: str = "max_length",
    ) -> tuple[torch.Tensor, list[int]]:
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
        return embeddings, inputs

    def get_embeddings(
        self,
        text_list: list[str],
        max_length: int = 128,
        truncation: bool = True,
        padding: str = "max_length",
        add_gaussian_noise: bool = False,
        noise_lambda: float = 0.1,
    ) -> tuple[torch.Tensor, list[int]]:

        if isinstance(self._encoder, OpenAIEncoder):
            embeddings = self._encoder(text_list)
            inputs = self._tokenizer(
                text_list,
                return_tensors="pt",
                max_length=max_length,
                truncation=truncation,
                padding=padding,
            )
            token_ids = inputs["input_ids"].tolist()
        else:
            embeddings, inputs = self._get_non_openai_embeddings(
                text_list, max_length=max_length, truncation=truncation, padding=padding
            )
            token_ids: list[int] = inputs["input_ids"].tolist()

        with torch.no_grad():
            if add_gaussian_noise:
                embeddings += noise_lambda * torch.normal(mean=0, std=1, size=embeddings.size())

        if self._cuda_is_available():
            embeddings = embeddings.to("cuda")

        return embeddings, token_ids

    def invert_embeddings(
        self,
        embeddings: torch.Tensor,
        num_steps: int = 0,
        min_length: int = 1,
        max_length: int = 32,
        sequence_beam_width: int = 0,
        do_sample: bool = False,
        top_p: float | None = None,
    ) -> list[str]:

        if self._cuda_is_available():
            embeddings = embeddings.to("cuda")
        else:
            embeddings = embeddings.to(self._corrector.model.device)

        self._corrector.inversion_trainer.model.eval()
        self._corrector.model.eval()

        gen_kwargs = copy.copy(self._corrector.gen_kwargs)
        gen_kwargs["min_length"] = min_length
        gen_kwargs["max_length"] = max_length
        gen_kwargs["do_sample"] = do_sample
        gen_kwargs["top_p"] = top_p

        if num_steps == 0:
            assert (
                sequence_beam_width == 0
            ), "can't set a nonzero beam width without multiple steps"

            outputs = self._corrector.inversion_trainer.generate(
                inputs={
                    "frozen_embeddings": embeddings,
                },
                generation_kwargs=gen_kwargs,
            )
        else:
            self._corrector.return_best_hypothesis = sequence_beam_width > 0
            outputs = self._corrector.generate(
                inputs={
                    "frozen_embeddings": embeddings,
                },
                generation_kwargs=gen_kwargs,
                num_recursive_steps=num_steps,
                sequence_beam_width=sequence_beam_width,
            )

        prediction_strs: list[str] = self._tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return prediction_strs

    def batch_encode_plus(self, text_list: list[str]) -> dict:
        out: dict = self._tokenizer.batch_encode_plus(text_list, return_tensors="pt", padding=True)
        return out

    def batch_decode(self, input_ids: list[int]) -> list[str]:
        out: list[str] = self._tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        return out
