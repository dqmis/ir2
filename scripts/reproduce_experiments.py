# Copied from
# https://github.com/jxmorris12/vec2text/blob/master/README.md#evaluate-the-models-from-the-papers
# Fully reproduces the evaluation of the model from the paper

from vec2text import analyze_utils

experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
    "jxm/gtr__nq__32__correct"
)
train_datasets = experiment._load_train_dataset_uncached(
    model=trainer.model, tokenizer=trainer.tokenizer, embedder_tokenizer=trainer.embedder_tokenizer
)

val_datasets = experiment._load_val_datasets_uncached(
    model=trainer.model, tokenizer=trainer.tokenizer, embedder_tokenizer=trainer.embedder_tokenizer
)
trainer.args.per_device_eval_batch_size = 16
trainer.sequence_beam_width = 1
trainer.num_gen_recursive_steps = 20
trainer.evaluate(eval_dataset=train_datasets["validation"])
