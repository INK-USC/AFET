#!/bin/sh
Data='BBN'
Indir='Data/'$Data
Intermediate='Intermediate/'$Data
Outdir='Results/'$Data
### Make intermediate and output dirs
mkdir -pv $Intermediate
mkdir -pv $Outdir

### Generate features
echo 'Step 1 Generate Features'
python DataProcessor/feature_generation.py $Data 20 
echo ' '

### Train AFET
echo 'Step 2 Train AFET'
Model/pl_warp $Data 50 0.01 50 10 0.15 0 1 1 5 1
echo ' '

### Predict and evaluate
echo 'Step 3 Predict and Evaluate'
python Evaluation/emb_prediction.py $Data pl_warp bipartite maximum cosine 0.25
python Evaluation/evaluation.py $Data pl_warp bipartite
