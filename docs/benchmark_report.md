# Pilot benchmark report

This report records the latest run completed in this workspace. The benchmark is small and synthetic; it demonstrates the evaluation path, not general natural-language understanding.

| Item | Measured result |
|---|---|
| Dataset version | `ddl-v2` |
| Accepted records | 104 |
| Train / validation / held-out generation test | 78 / 15 / 11 |
| Separate unseen-domain intent test | 15 cases |
| Rejected quality-control records | 4 |
| Scratch architecture | 145,152 parameters; 64 hidden size; 2 layers; 4 heads; 192-token context |
| Initialization | Random; `pretrained_weights_loaded: false` |
| Tokenizer | Corpus-trained byte-BPE; vocabulary 512 |
| Pilot device | CPU |
| Pilot training | 20 epochs; 16.54 seconds; 14,510 training tokens/second |
| Final train loss / perplexity | 2.272 / 9.70 |
| Best validation loss / perplexity | 2.728 / 15.30 |
| Hybrid scope-gate accuracy | 100% on 12 unseen-domain intent cases |
| Independent classifier accuracy | 73.33% on the same 15 cases |
| Independent classifier weakest class | `NEEDS_CLARIFICATION` F1 = 0% on this tiny sample |
| Scratch held-out syntax validity | 0% on 4 in-scope generation cases |
| Scratch normalized exact match | 0% |
| Ollama baseline | Blocked: local Ollama/model was unavailable |

The hybrid gate is intentionally conservative and combines the independently trained classifier with deterministic operation and safety policies. Its 100% result is not evidence of universal generalization; the benchmark must grow substantially and include human-authored unseen requests. The independent classifier's 75% result and clarification errors are retained as the actual limitation.

The scratch pilot is a verified random-initialized training run but is not production-ready: it fails the proposed 95% syntax and 90% semantic DDL targets. The next model experiment should improve target serialization, train on more valid DDL variation, and rerun the identical held-out generation suite.

The Ollama result is recorded as unavailable, not as a score. Run `python -m training.ollama_baseline --variant minimal` and `--variant common` after installing and explicitly pulling the selected model to complete the comparison.
