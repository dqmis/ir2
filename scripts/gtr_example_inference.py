import torch
import vec2text
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


# For sanity checks, just a copy of inference examples
# https://github.com/jxmorris12/vec2text/blob/master/README.md
def get_gtr_embeddings(
    text_list, encoder: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> torch.Tensor:

    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    ).to("cuda")

    with torch.no_grad():
        model_output = encoder(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs["attention_mask"])

    return embeddings


encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
corrector = vec2text.load_pretrained_corrector("gtr-base")

embeddings = get_gtr_embeddings(
    [
        """
       In accounting, minority interest (or non-controlling interest) is
       the portion of a subsidiary corporation's stock that is not owned by the
       parent corporation. The magnitude of the minority interest in the subsidiary
       company is generally less than 50% of outstanding shares, or the corporation
       would generally cease to be a subsidiary of the parent.[1]
       """
    ],
    encoder,
    tokenizer,
)

result = vec2text.invert_embeddings(
    embeddings=embeddings.cuda(),
    corrector=corrector,
    num_steps=20,
)

print(result)
