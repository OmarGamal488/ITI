# P2P: From Post To Personality — MBTI Prediction from Social Media

Reproduction of **Ma et al. (CIKM 2025)** — *From Post To Personality: Harnessing LLMs for MBTI Prediction in Social Media*

**Author:** Omar Gamal ElKady | ITI - AI Track, Intake 46

## Overview

P2P is a dual-LLM framework that predicts Myers-Briggs personality types (MBTI) from social media posts. The pipeline uses:

1. **Fine-tuned Local LLM** (DeepSeek-R1-8B + QLoRA) — extracts personality features from raw posts
2. **RAG with FAISS** — retrieves k=5 most similar labeled users as in-context demonstrations
3. **Online LLM** (DeepSeek-V3 API) — makes the final 4-letter MBTI prediction using all evidence

```
User Posts --> [Fine-tuned Local LLM] --> Personality Features
                                                |
                                    Sentence-BERT + Hidden States
                                                |
                                         FAISS (k=5 NN)
                                                |
                    Posts + Features + 5 Similar Users
                                                |
                                     [DeepSeek-V3 API] --> MBTI Type
```

## Results

Per-dimension accuracy on the PersonalityCafe test set (1,735 users):

| Approach | I/E | N/S | T/F | J/P | Avg |
|---|---|---|---|---|---|
| Naive Bayes | 0.7689 | 0.8617 | 0.7389 | 0.6115 | 0.7452 |
| Logistic Regression | 0.7758 | 0.8617 | 0.8006 | 0.6807 | 0.7797 |
| SVM (RBF) | 0.7700 | 0.8646 | 0.7695 | 0.6778 | 0.7705 |
| XGBoost | 0.7746 | 0.8588 | 0.7337 | 0.6369 | 0.7510 |
| **P2P (ours)** | **0.7816** | **0.8628** | **0.7562** | **0.6305** | **0.7578** |
| P2P (paper) | 0.9321 | 0.9475 | 0.9306 | 0.8858 | 0.9240 |

## Dataset

[PersonalityCafe](https://www.kaggle.com/datasets/datasnaek/mbti-type/data) forum dataset:
- 8,675 users with self-reported MBTI labels
- 50 most recent posts per user
- 16 MBTI types with severe class imbalance (INFP: 1,832 vs ESTJ: 39)
- Split: 60% train / 20% validation / 20% test (stratified)

## Project Structure

```
.
├── notebooks/
│   ├── P2P_MBTI.ipynb              # Evaluation notebook (run this to see results)
│   └── P2P_Full_Pipeline.ipynb     # Full training + inference pipeline (needs GPU)
├── deployment/
│   ├── app.py                      # Gradio demo app
│   ├── requirements.txt
│   └── README.md                   # HF Spaces metadata
├── Resources/
│   ├── 2509.04461v1.pdf            # P2P paper (Ma et al., CIKM 2025)
│   ├── 2025.coling-main.339.pdf    # MBTIBench paper
│   └── 2201.08717v1.pdf            # Ontoum 2022 paper
├── P2P_Implementation_Guide.pdf    # Step-by-step implementation guide
├── .gitignore
└── README.md
```

## Notebooks

### P2P_MBTI.ipynb (Evaluation)

Run this to see results. Loads pre-computed predictions from the full pipeline and displays:
- Per-dimension Accuracy, F1, AUC
- Baseline comparisons (Naive Bayes, LR, SVM, XGBoost)
- Confusion matrices (per-dimension + full 16-way)
- Prediction distribution analysis
- Live P2P pipeline demo (requires DeepSeek API key)

### P2P_Full_Pipeline.ipynb (Training)

Full reproduction of the paper. Requires a GPU with 16+ GB VRAM (e.g., Lightning AI, Kaggle T4x2). Covers:
1. Fine-tuning DeepSeek-R1-8B with QLoRA + SMOTE oversampling
2. Personality feature extraction via fine-tuned LLM
3. Hidden state extraction + FAISS vector database construction
4. RAG-augmented online prediction via DeepSeek-V3 API
5. Evaluation with baselines

## Links

- **Live Demo:** [HF Spaces](https://huggingface.co/spaces/OmarGamal48812/P2P-MBTI-Demo)
- **Fine-tuned Model:** [HF Model Hub](https://huggingface.co/OmarGamal48812/P2P-DeepSeek-R1-8B-MBTI-LoRA)
- **Paper:** Ma et al., *From Post To Personality: Harnessing LLMs for MBTI Prediction in Social Media*, CIKM 2025

## Setup

```bash
# Clone
git clone https://github.com/OmarGamal488/P2P-MBTI-Prediction.git
cd P2P-MBTI-Prediction

# Install dependencies
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install pandas pyarrow numpy scikit-learn xgboost matplotlib seaborn \
    sentence-transformers faiss-cpu openai python-dotenv

# Add API key (for live demo section)
echo 'DEEPSEEK_API_KEY=your-key-here' > .env

# Run evaluation notebook
jupyter lab notebooks/P2P_MBTI.ipynb
```

## References

```bibtex
@inproceedings{ma2025p2p,
  title={From Post To Personality: Harnessing LLMs for MBTI Prediction in Social Media},
  author={Ma, Tian and Feng, Kaiyu and Rong, Yu and Zhao, Kangfei},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  year={2025},
  publisher={ACM}
}
```
