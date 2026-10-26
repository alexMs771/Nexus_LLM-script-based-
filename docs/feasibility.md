# Feasibility and experiment contract

The supported hardware profile is Windows with 16 GB RAM, about 3.2 GB free RAM, and CPU-only execution. The default pilot is intentionally small: a 64-dimensional, two-layer, four-head decoder-only Transformer with a 192-token context and a byte-BPE tokenizer. It is a learning experiment, not a production-quality general model.

Intent classification is a separate five-class component. A character n-gram classifier is trained from labeled requests, while deterministic rules are reserved for clear completeness, unsupported-operation, and destructive-safety cases. The held-out intent benchmark uses domains not represented in the same form in training; its metrics are reported separately from the generator.

The scratch model is initialized by `SQLTransformer` and never reads pretrained weights. Its checkpoint contains the architecture, tokenizer, seed, parameter count, training/validation loss and perplexity, device, throughput, and `pretrained_weights_loaded: false` metadata. Run `python -m training.train_scratch --out-dir artifacts/runs/pilot` before increasing model size.

The byte tokenizer is a lossless baseline. The byte-BPE tokenizer learns merges from the training split only and falls back to bytes for unseen names. Compare the two using `python -m training.tokenizer_compare`.

Targets are gates, not claims: 95% syntax validity, 90% semantic correctness, and 95% intent F1 require a sufficiently large held-out benchmark. Results are written only after an actual run; no score is inferred from training loss.
