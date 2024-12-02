# Summary of progress from Serghei

Tried to reproduce the results from vec2text in 2 ways:

1.  Using the code provided in the README.md to load and run the experiment for sequence-length 32 GTR on the `nq` dataset with 20 steps - that worked, obtained the same results as in the paper.

```python
analyze_utils.load_experiment_and_trainer_from_pretrained(
	"jxm/gtr__nq__32__correct"
)
```

- it is based on this config: https://huggingface.co/jxm/gtr__nq__32__correct/blob/main/config.json - embedder: `"embedder_model_name": "gtr_base",` -> `sentence-transformers/gtr-t5-base` - corrector: `"corrector_model_from_pretrained": "jxm/gtr__nq__32",` ->

```python
inversion_model = vec2text.models.InversionModel.from_pretrained(
			            "jxm/gtr__nq__32"
			        )
model = vec2text.models.CorrectorEncoderModel.from_pretrained(
			            "jxm/gtr__nq__32__correct"
        )
```

2.  Using their example code for inference from README.md, and applying `trainer._text_comparison_metrics()` to the results - yielded worse results than in the paper - bleu 11, f1 0.4, 0 exact matches, on the same `nq` dataset. Qualitatively, the reproduced texts are similar, but still very far from matching, see examples below.
    - embedder: `sentence-transformers/gtr-t5-base`
    - corrector:
      - `self._corrector = vec2text.load_pretrained_corrector("gtr-base")` - this uses exactly the same models as in the experiments loaded in approach 1.
    - Thus approach 2 seems to use the same configuration as approach 1
    - Examples:
      - **Original**: "In accounting, minority interest (or non-controlling interest) is the portion of a subsidiary corporation's stock that is not owned by the parent corporation. The magnitude of the minority interest in the subsidiary company is generally less than 50% of outstanding shares, or the corporation would generally cease to be a subsidiary of the parent.[1]"
      - **With 100 steps**: "(a) minority interest is the non-acquiring portion of the shares that the parent corporation holds. In accounting for a corporation\'s"
      - **With 5 steps**: (and non minority interest is the portion of the controlling corporation's shares that the parent corporation does not own.) In accounting for such a corporation"
