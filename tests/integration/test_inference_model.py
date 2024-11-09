from ir2.inference_model import Vec2textInferenceModel

if __name__ == "__main__":
    model = Vec2textInferenceModel(
        model_name="sentence-transformers/gtr-t5-base", corrector_name="gtr-base"
    )

    INPUT = (
        [
            "Jack Morris is a PhD student at Cornell Tech in New York City",
            "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity",  # noqa
        ],
    )

    embeddings = model.get_embeddings(INPUT)
    inverted_embeddings = model.invert_embeddings(embeddings, num_steps=10)

    print(inverted_embeddings)
