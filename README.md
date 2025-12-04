# PyTorch Transformer (Seq2Seq)

Lightweight, educational Transformer-based sequence-to-sequence example using HuggingFace datasets and the `tokenizers` library. The project demonstrates training a small encoder-decoder Transformer for machine translation (example: English ↔ Italian) and includes simple training and inference scripts.

**Status**: Minimal working training & inference scripts. Intended for learning and experimentation.

---

## Quickstart

- **Install dependencies**

```bash
pip install -r requirements.txt
```

- **Train** (uses settings from `config.py`):

```bash
python train.py
```

- **Translate a sentence or dataset index** (inference uses latest saved weights):

```bash
python translate.py "This is a test sentence."
# or
python translate.py 12
```

Notes:
- Tokenizers are created automatically and saved using the pattern `tokenizer_{lang}.json`.
- Weights are saved to a folder named using the datasource and model folder configured in `config.py` (example: `opus_en-it_weights`).

---

## Files & Key Components

- `config.py` — configuration helpers and utility functions (e.g. `get_config`, `get_weights_file_path`, `latest_weights_file_path`).
- `dataset.py` — dataset wrapper and helper masks (`BilingualDataset`, `causal_mask`).
- `model.py` — Transformer implementation and `build_transformer` factory.
- `train.py` — training pipeline, tokenizer building/loading, evaluation helpers, and model checkpointing.
- `translate.py` — simple script to perform greedy decoding / translation with saved weights.
- `requirements.txt` — Python dependencies used by the project.

---

## Configuration

Open `config.py` to change dataset, languages, model and training hyperparameters. Typical options you will find there:

- `datasource`: the HuggingFace dataset id used (e.g. `opus100` or another dataset).
- `lang_src`, `lang_tgt`: source and target language codes.
- `seq_len`, `d_model`, `batch_size`, `num_epochs`, `lr`: model and training hyperparameters.
- `model_folder`: folder name suffix used when saving weights. Weights are stored under `<datasource>_<model_folder>`.

---

## Tokenizers

`train.py` creates tokenizers using `tokenizers.WordLevel` when they don’t already exist. Files are saved according to the pattern defined in `config.py` (normally `tokenizer_{lang}.json`). If you want to replace or inspect a tokenizer, open the JSON file saved by the `tokenizers` library.

---

## Checkpoints & Inference

- Checkpoints saved by `train.py` include the model state, optimizer state, epoch and `global_step`.
- `translate.py` looks for the latest saved weights via the helper in `config.py` (see `latest_weights_file_path`).

---

## Common Commands

- Run training and stream logs to TensorBoard:

```bash
python train.py
tensorboard --logdir runs --bind_all
```

- Run a quick translation (inference):

```bash
python translate.py "I am learning transformers."
```

---

## Troubleshooting

- GPU not used: check PyTorch CUDA installation and the device selection message printed by `train.py`.
- Tokenizer issues: delete the `tokenizer_{lang}.json` files to force rebuilding (useful after changing tokenization settings).
- Out of memory: reduce `batch_size` or `seq_len`, or use a smaller `d_model`.

---

## Next steps / Suggestions

- Add unit tests or a small example script that loads a checkpoint and runs inference deterministically.
- Add configurable learning-rate schedulers and checkpoint rotation.
- Replace `tokenizers.WordLevel` with a BPE/WordPiece tokenizer for better generalization.

---

## License

This repository does not include an explicit license file. Add a `LICENSE` file if you plan to publish or share this project.

If you want me to expand any section (example configs, reproduceable experiment commands, or a minimal test harness), tell me which part to add.
