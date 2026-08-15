# Neural Chunking with BiLSTM and Transformer Encoders

This project labels the role of every token in a sentence using the BIO format. For example, `New` can begin a noun phrase with `B-NP` and `York` can continue it with `I-NP`. It is a compact sequence-labelling pipeline built around the parts that make token classification reliable: sentence-aware splitting, padding-safe neural models, and both token and exact-span evaluation.

The package implements a bidirectional LSTM and a compact Transformer behind the same training and evaluation interface. The retained metrics below are for the BiLSTM only. No Transformer metrics were preserved, so the two architectures are not presented as an empirical comparison.

[Open the saved project walkthrough](notebooks/neural_chunking_walkthrough.ipynb) to see the result table, training curves, confusion matrix, and direct links to the main implementation.

## Results

The stored BiLSTM evaluation selected epoch 16 and scored 31,833 tokens from 1,339 held-out sentences.

| Metric | Score |
|---|---:|
| Token accuracy | 0.9430 |
| Token macro F1 | 0.7521 |
| Token weighted F1 | 0.9423 |
| Matthews correlation coefficient | 0.9288 |
| Loss | 0.2158 |

Macro F1 is lower than weighted F1 because it gives rare BIO labels the same importance as frequent labels. That makes it a useful check that strong performance is not confined to the common `O` label.

![BiLSTM confusion matrix](artifacts/figures/best_bilstm_confusion_matrix.png)

![BiLSTM token accuracy by epoch](artifacts/figures/bilstm_accuracy_curve.png)

![BiLSTM loss by epoch](artifacts/figures/bilstm_loss_curve.png)

The figures and [stored metrics](artifacts/results/retained_metrics.json) are integrity-pinned in [`artifacts/SHA256SUMS`](artifacts/SHA256SUMS). Verify them with `shasum -a 256 -c artifacts/SHA256SUMS`.

## How it works

1. Read token and BIO-label pairs, preserving sentence boundaries.
2. Keep exact duplicate sentences together when splitting the data, then build token and label vocabularies from the training partition only.
3. Batch variable-length sentences with masks so padding cannot affect recurrent states, attention, loss, decoded labels, or metrics.
4. Train either encoder, choose a checkpoint from validation evidence, then evaluate that selected checkpoint once on the held-out partition.
5. Report token accuracy, F1, Matthews correlation, and exact BIO-span precision, recall, and F1 for new runs.

## Skills demonstrated

| Area | Where to look |
|---|---|
| Data validation, duplicate-aware splitting, vocabulary construction, and batching | [`data.py`](src/neural_chunking/data.py) |
| PyTorch BiLSTM and Transformer sequence encoders | [`models.py`](src/neural_chunking/models.py) |
| BIO decoding and exact-span metrics | [`metrics.py`](src/neural_chunking/metrics.py) |
| Reproducible training, checkpoint selection, and evaluation safeguards | [`training.py`](src/neural_chunking/training.py) |
| Command-line training and evaluation workflow | [`cli.py`](src/neural_chunking/cli.py) |

## Run it

The input is a UTF-8 text file containing one `token BIO-label` pair per line, with blank lines between sentences. Keep source data outside version control and use a corpus whose licence allows the intended use.

```text
New B-NP
York I-NP
works B-VP

```

```bash
uv sync --extra test
uv run neural-chunking train /path/to/chunks.txt --architecture bilstm --output runs/bilstm
uv run neural-chunking evaluate /path/to/chunks.txt runs/bilstm/chunker_checkpoint.pt --output runs/bilstm/test-results.json
```

To run the Transformer, change `--architecture bilstm` to `--architecture transformer`. Its positional encoding supports sentences up to 512 tokens by default. The BiLSTM has no positional-encoding limit.

## Limitations

- The source corpus name, version, hash, and licence were not retained with the stored metrics, so this repository does not claim a dataset-specific benchmark.
- Predictions from that BiLSTM run are unavailable. Exact-span F1 cannot be recovered and is therefore not reported above.
- Transformer metrics were not retained.
- Content hashing keeps exact duplicate sentences together, but it cannot identify paraphrases or source-level groups that are not represented in the input.
- Neither encoder uses pretrained language representations.
