import sys
from collections import  defaultdict

def evaluate(prediction, ground_truth):
    """
    Evaluation matrix.
    :param prediction: a dictionary of labels. e.g {0:[1,0],1:[2],2:[3,4],3:[5,6,7]}
    :param ground_truth: a dictionary of labels
    :return:
    """
    print "prediction:%d, ground:%d"%(len(prediction),len(ground_truth))
    assert len(prediction) == len(ground_truth)
    count = len(prediction)
    # print 'Test', count, 'mentions'
    same = 0
    macro_precision = 0.0
    macro_recall = 0.0
    micro_n = 0.0
    micro_precision = 0.0
    micro_recall = 0.0

    for i in ground_truth:
        p = prediction[i]
        g = ground_truth[i]
        if p == g:
            same += 1
        same_count = len(p&g)
        macro_precision += float(same_count)/float(len(p))
        macro_recall += float(same_count)/float(len(g))
        micro_n += same_count
        micro_precision += len(p)
        micro_recall += len(g)

    accuracy = float(same) / float(count)
    macro_precision /= count
    macro_recall /= count
    macro_f1 = 2*macro_precision*macro_recall/(macro_precision + macro_recall + 1e-8)
    micro_precision = micro_n/micro_precision
    micro_recall = micro_n/micro_recall
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    return accuracy,macro_precision,macro_recall,macro_f1,micro_precision,micro_recall,micro_f1


def load_labels(file_name):
    labels = defaultdict(set)
    with open(file_name) as f:
        for line in f:
            seg = line.strip('\r\n').split('\t')
            labels[int(seg[0])].add(int(seg[1]))
        f.close()
    return labels

def load_raw_labels(file_name, ground_truth):
    labels = defaultdict(set)
    with open(file_name) as f:
        for line in f:
            seg = line.strip('\r\n').split('\t')
            if int(seg[0]) in ground_truth:
                labels[int(seg[0])].add(int(seg[1]))
        f.close()
    return labels

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print 'Usage: evaluation.py -DATA(BBN) \
         -METHOD(pl_warp) -EMB_MODE(bipartite)'
        exit(-1)
    _data = sys.argv[1]
    _method = sys.argv[2]
    _emb_mode = sys.argv[3] 
    ground_truth = load_labels('Intermediate/' + _data + '/mention_type_test.txt')

    ### Evluate embedding predictions
    predictions = load_labels('Results/' + _data + '/mention_type_' + _method + '_' + _emb_mode + '.txt')
    print 'Predicted labels (embedding):'
    accuracy,macro_precision,macro_recall,macro_f1,micro_precision,micro_recall,micro_f1 = evaluate(predictions, ground_truth)
    print 'accuracy:', accuracy
    print 'macro_precision, macro_recall, macro_f1:', macro_precision, macro_recall, macro_f1
    print 'micro_precision, micro_recall, micro_f1:', micro_precision, micro_recall, micro_f1
