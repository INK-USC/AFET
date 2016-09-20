Please download [BBN](https://drive.google.com/file/d/0B2ke42d0kYFfdVk2ZkJ6TGRzR2M/view?usp=sharing) and unzip the file here.

Runing paramter:
```
Model/pl_warp BBN 50 0.01 50 10 0.25 3 1 1 5 1
python Evaluation/emb_prediction.py BBN pl_warp bipartite maximum cosine 0.12
```
