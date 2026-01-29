# AI-Guided Protein Feature Engineering

## Motivation
Protein engineering is traditionally slow and relies on trial-and-error experimentation. Inspired by AI-driven approaches such as AiCE, this project explores how protein sequences can be transformed into numerical features suitable for machine learning and how different representations influence predictive performance.

## Methods
- Cleaned and processed real protein sequence data
- Extracted physicochemical features using Biopython ProtParam
- Computed amino acid composition (AAC) vectors
- Trained Random Forest classifiers to predict protein class labels
- Compared model performance across feature representations
- Simulated single–amino-acid mutations to analyze downstream effects on protein properties

## Results
- ProtParam features accuracy: ~0.57
- Amino acid composition (AAC) accuracy: ~0.59
- Mutation simulations revealed position-specific effects on predicted protein stability

## Connection to AiCE
This project mirrors foundational steps in AI-guided protein engineering frameworks such as AiCE, where protein representations and mutation effects are analyzed prior to optimization and experimental validation.

## Future Work
- Structure-aware protein embeddings
- Mutation ranking and optimization
- Deep learning models for sequence-to-function prediction
