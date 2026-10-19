# Q-VAIDYA

**QLoRA-based 8-bit quantization and fine-tuning of Llama-4 Scout (MoE) for clinical decision support.**

Q-VAIDYA compresses the 109B-parameter Llama-4 Scout Mixture-of-Experts model to an 8-bit backbone and fine-tunes it with low-rank adapters on medical question-answering data. It roughly halves the memory footprint and keeps about 96% of full-precision accuracy on PubMedQA and MedQA-USMLE.

<p align="center">
  <code>Medical data (text + images)</code> → <code>Llama-4 Scout (BF16)</code> → <code>8-bit quantization + QLoRA</code> → <code>Q-VAIDYA</code>
</p>

---

## Highlights

- **Memory:** about 50% less VRAM than full precision (218 GB → 109 GB during fine-tuning).
- **Parameter-efficient:** about 94M trainable LoRA parameters (under 0.1% of the model); each adapter checkpoint is about 180 MB.
- **Accuracy retained:** 72.8% on PubMedQA and 71.1% on MedQA-USMLE, against 73% and 72% for the BF16 baseline.
- **Calibrated outputs:** constrained decoding gives a proper probability distribution over answers, scored with ECE and Brier score.
- **Multimodal:** text and medical-image samples share one instruction-masked training pipeline.

## Repository structure

```
LLAMA-4/
├── configs/
│   ├── train_config.yaml      # model, LoRA, data and optimiser settings
│   └── eval_config.yaml       # benchmarks and evaluation settings
├── scripts/
│   ├── train.sh               # fine-tune with 8-bit QLoRA
│   └── evaluate.sh            # evaluate BF16 baseline and the 8-bit QLoRA model
├── src/qvaidya/
│   ├── data.py                # dataset loading, chat formatting, instruction masking, split
│   ├── model.py               # 8-bit loading, frozen backbone, LoRA injection
│   ├── router_patch.py        # MoE router compatibility patch
│   ├── train.py               # training entry point (HF Trainer)
│   ├── evaluate.py            # constrained-decoding and VQA evaluation
│   └── metrics.py             # accuracy, F1, MCC, ECE, Brier, ROUGE, BERTScore
├── tests/
│   └── test_metrics.py
├── pyproject.toml
└── requirements.txt
```

## Method

### 1. Data preparation

| Dataset | Samples used | Format |
|---|---|---|
| [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (`pqa_labeled`) | 1,000 | context + question → yes / no / maybe |
| [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | 1,000 | clinical vignette → A / B / C / D |
| [Awesome Medical Dataset](https://github.com/medicalai/awesome-medical-dataset) | 2,991 | medical image + question → free-text answer |

Each sample is rendered with the Llama-4 chat template and tokenized to at most 2,048 tokens with right padding. Labels for every prompt and padding position are set to `-100`, so the cross-entropy loss covers only the assistant's answer. The combined data is split 95/5 into train and validation sets, stratified by source and answer label.

### 2. 8-bit QLoRA training

| Setting | Value |
|---|---|
| Base model | `meta-llama/Llama-4-Scout-17B-16E-Instruct` (48 layers, 16 routed experts + 1 shared, top-1 routing) |
| Quantization | bitsandbytes 8-bit, `llm_int8_threshold=6.0`, frozen backbone, router and LM head kept in BF16 |
| LoRA | `r=64`, `alpha=128`, `dropout=0.05` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimiser | paged AdamW 8-bit, lr `2e-4`, 3% linear warmup, cosine decay |
| Batch | 2 per device × 16 accumulation steps |
| Stability | gradient checkpointing, gradient clipping at 0.3, CUDA cache cleared every epoch |
| Epochs | 10, best checkpoint chosen by validation loss |
| Hardware | 8× NVIDIA H100 80GB |

> **Note:** the paper calls the backbone format FP8. The code uses bitsandbytes `load_in_8bit`, which stores weights in 8-bit (LLM.int8) and dequantizes them to BF16 during the forward pass.

Before the model loads, a small patch (`router_patch.py`) makes the MoE router unpack its outputs defensively. Without it, the forward pass can break on `transformers` versions that return a tuple instead of a tensor.

### 3. Evaluation

- **PubMedQA / MedQA-USMLE:** constrained decoding. The logits for the valid answer tokens (yes/no/maybe or A–D) are passed through a restricted softmax to get `P(a | x)`. Reported metrics are accuracy, macro and weighted F1, MCC, ECE and Brier score.
- **VQA-RAD-100:** greedy generation on 100 radiology image questions, scored with ROUGE-L, exact match and BERTScore F1.

## Results

**Medical reasoning benchmarks**

| Metric | PubMedQA BF16 | PubMedQA 8-bit QLoRA | MedQA BF16 | MedQA 8-bit QLoRA |
|---|---|---|---|---|
| Accuracy | 73.0% | 72.8% | 72.0% | 71.1% |
| Macro F1 | 0.5765 | 0.5540 | 0.7087 | 0.6815 |
| Weighted F1 | 0.7288 | 0.7003 | 0.7113 | 0.6840 |
| MCC | 0.5180 | 0.4978 | — | — |

**Awesome Medical Dataset (2,991 samples)**

| Metric | BF16 | 8-bit QLoRA |
|---|---|---|
| ROUGE-L | 0.7370 | 0.7121 |
| ROUGE-1 | 0.7371 | 0.7135 |
| ROUGE-2 | 0.7350 | 0.7110 |
| BERTScore F1 | 0.7700 | 0.7445 |
| Partial match | 0.9840 | 0.9512 |
| Keyword recall | 1.0000 | 0.9642 |

**VQA-RAD-100**

| Metric | BF16 | 8-bit QLoRA |
|---|---|---|
| ROUGE-L | 0.3840 | 0.3801 |
| Exact match | 0.3400 | 0.2700 |
| BERTScore F1 | 0.7028 | 0.6549 |

Text-based reasoning survives quantization almost unchanged. Visual question answering is more sensitive, with exact match down 7 points, which suggests mixed-precision strategies for radiology use.

**Fine-tuning memory**

| Method | VRAM (GB) |
|---|---|
| Full precision | 218 |
| LoRA (BF16) | 140 |
| AdaLoRA (BF16) | 145 |
| **8-bit QLoRA (Q-VAIDYA)** | **109** |

## Getting started

### Requirements

- Python 3.10+
- CUDA GPUs with roughly 110 GB of combined VRAM for the 8-bit backbone (the model is sharded with `device_map="auto"`)
- Access to the gated [Llama-4 Scout](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) weights on Hugging Face

```bash
git clone https://github.com/Adityacrypto-del/LLAMA-4.git
cd LLAMA-4
pip install -r requirements.txt
pip install -e .
huggingface-cli login
```

### Data

PubMedQA, MedQA-USMLE and VQA-RAD download automatically from the Hugging Face Hub. The Awesome Medical Dataset is a collection rather than a single Hub dataset. Put your prepared copy in `data/awesome_medical`, either as a saved HF dataset or as a JSON/JSONL file with `question`, `answer` and an optional `image` path, or point `data.sources.awesome_medical.path` in `configs/train_config.yaml` somewhere else. If the path doesn't exist, that source is skipped.

### Train

```bash
bash scripts/train.sh                        # uses configs/train_config.yaml
```

Checkpoints are written to `outputs/q-vaidya/`, and the best adapter goes to `outputs/q-vaidya/best_adapter/`.

### Evaluate

```bash
bash scripts/evaluate.sh                     # BF16 baseline + 8-bit QLoRA

# or run one configuration
PYTHONPATH=src python -m qvaidya.evaluate --config configs/eval_config.yaml
PYTHONPATH=src python -m qvaidya.evaluate --config configs/eval_config.yaml --no-adapter --bf16
```

Results are saved as JSON in `outputs/eval/`.

### Tests

```bash
pytest
```

## Authors

Aditi Choudhary, Sanika Madihal, Sathvik Kemtur, Aditya Arasamangalam, and Dr. Uday Kulkarni
School of Computer Science and Engineering, KLE Technological University, Hubballi, India

## Citation

```bibtex
@article{qvaidya2025,
  title   = {Q-VAIDYA: QLoRA-Based FP8 Quantization and Optimization of Llama-4 MoE for Clinical Decision Support},
  author  = {Choudhary, Aditi and Madihal, Sanika and Kemtur, Sathvik and Arasamangalam, Aditya and Kulkarni, Uday},
  journal = {Preprint submitted to Elsevier},
  year    = {2025}
}
```

## Disclaimer

This is a research project. It is not a medical device and must not be used for clinical decisions without qualified medical oversight. Use of Llama-4 is subject to the [Llama 4 Community License](https://www.llama.com/llama4/license/).
