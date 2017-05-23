__author__ = 'wenqihe'
import sys
import math
from multiprocessing import Process, Lock
from nlp_parse import parse
from ner_feature import pipeline, filter, pipeline_test
from statistic import supertype, distribution
from pruning_heuristics import prune
from type_type_kb import share_entity

def get_number(filename):
    with open(filename) as f:
        count = 0
        for line in f:
            count += 1
        return count

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'Usage:feature_generation.py -DATA -numOfProcesses'
        exit(1)
    indir = 'Data/%s' % sys.argv[1]
    outdir = 'Intermediate/%s' % sys.argv[1]
    # NLP parse
    raw_train_json = indir + '/train.json'
    raw_test_json = indir + '/test.json'
    train_json = outdir + '/train_new.json'
    test_json = outdir + '/test_new.json'

    # Generate features
    print 'Start nlp parsing'
    file = open(raw_train_json, 'r')
    sentences = file.readlines()
    numOfProcesses = int(sys.argv[2])
    sentsPerProc = int(math.floor(len(sentences)*1.0/numOfProcesses))
    lock = Lock()
    processes = []
    train_json_file = open(train_json, 'w', 0)
    for i in range(numOfProcesses):
        if i == numOfProcesses - 1:
            p = Process(target=parse, args=(sentences[i*sentsPerProc:], train_json_file, lock))
        else:
            p = Process(target=parse, args=(sentences[i*sentsPerProc:(i+1)*sentsPerProc], train_json_file, lock))
        p.start()
        processes.append(p)
    for proc in processes:
        proc.join()
    train_json_file.close()

    file = open(raw_test_json, 'r')
    numOfProcesses = int(sys.argv[2])
    sentences = file.readlines()
    sentsPerProc = int(math.floor(len(sentences)*1.0/numOfProcesses))
    processes = []
    lock = Lock()
    test_json_file = open(test_json, 'w', 0)
    for i in range(numOfProcesses):
        if i == numOfProcesses - 1:
            p = Process(target=parse, args=(sentences[i*sentsPerProc:], test_json_file, lock))
        else:
            p = Process(target=parse, args=(sentences[i*sentsPerProc:(i+1)*sentsPerProc], test_json_file, lock))
        p.start()
        processes.append(p)
    for proc in processes:
        proc.join()
    test_json_file.close()
    print 'Start feature extraction'
    pipeline(train_json, indir + '/brown', outdir)
    filter(outdir+'/feature.map', outdir+'/train_x.txt', outdir+'/feature.txt', outdir+'/train_x_new.txt')
    pipeline_test(test_json, indir + '/brown', outdir+'/feature.txt',outdir+'/type.txt', outdir)
    supertype(outdir)
    distribution(outdir)

    # Perform no pruning to generate training data
    print 'Start training and test data generation'
    feature_number = get_number(outdir+'/feature.txt')
    type_number = get_number(outdir+'/type.txt')
    prune(outdir, outdir, 'no', feature_number, type_number)

    # Generate type type correlation
    print 'Start type correlation calculation'
    share_entity(indir + '/type_entities.txt', outdir + '/type.txt', outdir + '/type_type_kb.txt')
