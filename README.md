## Requirements

We will take Ubuntu for example.

* python 2.7

```
$ sudo apt-get install python
```

* [stanford coreNLP 3.5.2](http://stanfordnlp.github.io/CoreNLP/) and its [python wrapper](https://github.com/dnc1994/stanford-corenlp-python). Please put the library in folder DataProcessor/.

```
$ cd DataProcessor/
$ sudo pip install pexpect unidecode
$ git clone git://github.com/dnc1994/stanford-corenlp-python.git
$ cd stanford-corenlp-python
$ python setup.py install
$ cd corenlp
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
$ unzip stanford-corenlp-full-2015-04-20.zip
```

## Build 
Build our model in folder Model.

```
$ cd Model/
$ make
```

## Dataset
Please put the data files in corresponding subdirectories in Data/. You could download [Wiki](https://drive.google.com/file/d/0B2ke42d0kYFfVC1fazdKYnVhYWs/view?usp=sharing), [OntoNotes](https://drive.google.com/file/d/0B2ke42d0kYFfN1ZSVExLNlYwX1E/view?usp=sharing), [BBN](https://drive.google.com/file/d/0B2ke42d0kYFfdVk2ZkJ6TGRzR2M/view?usp=sharing) in Google Drive.

## Default Run
Run AFET for fine-grained entity typing on BBN dataset

```
$ ./run.sh  
```

## Parameters - run.sh
Dataset to run on.
```
Data="BBN"
```

