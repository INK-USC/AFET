Please download [OntoNotes](https://drive.google.com/file/d/0B2ke42d0kYFfN1ZSVExLNlYwX1E/view?usp=sharing) and unzip the file here.

Runing paramter:
```
Model/pl_warp OntoNotes 50 0.003 50 10 0.15 0 1 1 5 0.7
python Evaluation/emb_prediction.py OntoNotes pl_warp bipartite maximum cosine 0.7
