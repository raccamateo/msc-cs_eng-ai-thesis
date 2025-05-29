# msc-cs_eng-ai-thesis

**Comparative Analysis of Transformer-Based LLMs and Spiking Neural Networks for Contract Clause Extraction**

Author: Mateo W. Racca  
M.Sc. in Computer Security Engineering  and Artificial Intelligence, URV

## Structure

- `data/`  
  - `download_datasets.py`: Downloads and unpacks contract datasets.  
  - `preprocess.py`: Cleans and tokenizes text.  
  - `annotate.py`: Aligns annotations to tokens.  
- `models/`  
  - `train_legalbert.py`: Fine-tunes LEGAL-BERT for clause extraction.  
  - `train_snn.py`: Implements a two-layer SNN with surrogate gradients.  
- `evaluation/`  
  - `evaluate_models.py`: Evaluates both models, outputs metrics and plots.  
- `utils/`  
  - `utils.py`: Shared helper functions.  

## Quickstart

```bash
conda env create -f environment.yml
conda activate thesis-env

# Data pipeline
python data/download_datasets.py
python data/preprocess.py
python data/annotate.py

# Train
python models/train_legalbert.py --output_dir outputs/legalbert
python models/train_snn.py --output_dir outputs/snn

# Evaluate
python evaluation/evaluate_models.py --legalbert_dir outputs/legalbert --snn_model outputs/snn/snn_model.pt
```
