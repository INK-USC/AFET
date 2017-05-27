## AFET: Automatic Fine-Grained Entity Typing by Hierarchical Partial-Label Embedding

Source code and data for EMNLP'16 paper *[AFET: Automatic Fine-Grained Entity Typing by Hierarchical Partial-Label Embedding](http://xren7.web.engr.illinois.edu/16-AFET.pdf)*. 

Given a text corpus with entity mentions *detected* and *heuristically labeled* by distant supervision, this code performs training of a rank-based loss over distant supervision and predict the fine-grained entity types for each test entity mention. For example, check out AFET's [output on WSJ news articles](https://raw.githubusercontent.com/shanzhenren/PLE/master/Results/BBN/predictionInText_hple_hete_feature_perceptron.txt).

An end-to-end tool (corpus to typed entities) is under development. Please keep track of our updates.

## Performance
Performance of *fine-grained entity type classification* over **Wiki** ([Ling & Weld, 2012](http://xiaoling.github.io/pubs/ling-aaai12.pdf)) dataset. We applied PLE to clean training data and ran FIGER ([Ling & Weld, 2012](http://xiaoling.github.io/pubs/ling-aaai12.pdf)) and over the de-noised labeled data to train type classifiers (thus the **FIGER + PLE** is the name of our final system).

Method | Accuray | Macro-F1 | Micro-F1 
-------|-----------|--------|----
HYENA ([Yosef et al., 2012](http://aclweb.org/anthology/C/C12/C12-2133.pdf)) | 0.288 | 0.528 | 0.506 
WSABIE ([Yogatama et al,., 2015](http://www.cs.cmu.edu/~dyogatam/papers/yogatama+etal.acl2015short.pdf)) | 0.480 | 0.679 | 0.657 
FIGER ([Ling & Weld, 2012](http://xiaoling.github.io/pubs/ling-aaai12.pdf)) | 0.474 | 0.692 | 0.655 
FIGER + All Filter ([Gillick et al., 2014](https://arxiv.org/pdf/1412.1820.pdf)) |0.453 | 0.648 | 0.582 
**FIGER + PLE** ([Ren et al., 2016](https://arxiv.org/pdf/1602.05307.pdf)) | **0.599** | **0.763** | **0.749**


## System Output
The output on [BBN dataset](https://drive.google.com/file/d/0B2ke42d0kYFfTEs0RGpuanRLQlE/view?usp=sharing) can be found [here](https://raw.githubusercontent.com/shanzhenren/PLE/master/Results/BBN/predictionInText_hple_hete_feature_perceptron.txt). Each line is a sentence in the test data of BBN, with entity mentions and their fine-grained entity typed identified.


## Dependency
* python 2.7, g++
* Python library dependencies
```
$ pip install pexpect unidecode six requests protobuf
```
* Setup [stanford coreNLP](http://stanfordnlp.github.io/CoreNLP/) and its [python wrapper](https://github.com/stanfordnlp/stanza).
```
$ cd DataProcessor/
$ git clone git@github.com:stanfordnlp/stanza.git
$ cd stanza
$ pip install -e .
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
$ unzip stanford-corenlp-full-2016-10-31.zip
$ rm stanford-corenlp-full-2016-10-31.zip
```


## Makefile
```
$ cd AFET/Model/ple/; make
```

## Dataset
Please put the data files in corresponding subdirectories in Data/. You could download [Wiki](https://drive.google.com/file/d/0B2ke42d0kYFfVC1fazdKYnVhYWs/view?usp=sharing), [OntoNotes](https://drive.google.com/file/d/0B2ke42d0kYFfN1ZSVExLNlYwX1E/view?usp=sharing), [BBN](https://drive.google.com/file/d/0B2ke42d0kYFfdVk2ZkJ6TGRzR2M/view?usp=sharing) in Google Drive.

## Default Run
Run AFET for fine-grained entity typing on BBN dataset

```
$ java -mx4g -cp "DataProcessor/stanford-corenlp-full-2016-10-31/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
$ ./run.sh  
```

## Parameters - run.sh
Dataset to run on.
```
Data="BBN"
```

## Publication
Please cite the following paper if you find the codes and datasets are helpful:
```
@inproceedings{Ren2016AFETAF,
  title={AFET: Automatic Fine-Grained Entity Typing by Hierarchical Partial-Label Embedding},
  author={Xiang Ren and Wenqi He and Meng Qu and Lifu Huang and Heng Ji and Jiawei Han},
  booktitle={EMNLP},
  year={2016}
}
```
