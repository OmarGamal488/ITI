---
title: P2P - From Post To Personality
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.23.0"
app_file: app.py
pinned: false
python_version: "3.11"
---

# P2P: From Post To Personality — MBTI Prediction

Predict Myers-Briggs personality type from social media posts using the P2P framework.

**Author:** Omar Gamal ElKady | ITI - AI Track, Intake 46

**Paper:** Ma et al. (CIKM 2025) — *From Post To Personality: Harnessing LLMs for MBTI Prediction in Social Media*

## How it works

1. **Feature Extraction**: DeepSeek-V3 analyzes your text and extracts personality features across the 4 MBTI dimensions (E/I, S/N, T/F, J/P)
2. **MBTI Prediction**: DeepSeek-V3 predicts your 4-letter MBTI type based on the extracted features

## Fine-tuned Model

The fine-tuned LoRA adapter is available at: [OmarGamal48812/P2P-DeepSeek-R1-8B-MBTI-LoRA](https://huggingface.co/OmarGamal48812/P2P-DeepSeek-R1-8B-MBTI-LoRA)

## Setup

Add your `DEEPSEEK_API_KEY` as a Space secret in Settings.
