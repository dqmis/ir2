import vec2text
from datasets import load_dataset
# corrector = vec2text.load_pretrained_corrector("gtr-base")

# result = vec2text.invert_strings(
#     [
#         "Jack Morris is a PhD student at Cornell Tech in New York City",
#         "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity",
#         "I like trains"
#     ],
#     corrector=corrector,
#     num_steps = 20,
# )

# print(result)

ds = load_dataset("BeIR/quora", "corpus")
print(ds["corpus"]["text"][0])