# qwen-finetune

```sh
pixi install
pixi run train
pixi run inference
```

Target roughly:
- 500 examples - noticeable behavior
- 2,000 examples - solid specialized behavior
- 5,000+ examples - serious fine-tune

Unsloth recommends approximately 75% reasoning examples + 25% non-reasoning examples if preserving reasoning is important.

Once you have your full dataset, replace data/train.jsonl with it and set SMOKE_TEST = False in train.py.
