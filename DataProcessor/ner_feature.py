
from Feature import *
import sys
from mention_reader import MentionReader
reload(sys)
sys.setdefaultencoding('utf8')

class NERFeature(object):

    def __init__(self, is_train, brown_file, feature_mapping={}, label_mapping={}):
        self.is_train = is_train
        self.feature_count = 0
        self.label_count = 0
        self.feature_list = []
        self.feature_mapping = feature_mapping # {feature_name: [feature_id, feature_frequency]}
        self.label_mapping = label_mapping # {label_name: [label_id, label_frequency]}
        # head feature
        self.feature_list.append(HeadFeature())
        # token feature
        self.feature_list.append(TokenFeature())
        # context unigram
        self.feature_list.append(ContextFeature(window_size=3))
        # context bigram
        self.feature_list.append(ContextGramFeature(window_size=3))
        # pos feature
        self.feature_list.append(PosFeature())
        # word shape feature
        self.feature_list.append(WordShapeFeature())
        # length feature
        self.feature_list.append(LengthFeature())
        # character feature
        self.feature_list.append(CharacterFeature())
        # brown clusters
        self.feature_list.append(BrownFeature(brown_file))
        # dependency feature
        self.feature_list.append(DependencyFeature())


    def extract(self, sentence, mention):
        # extract feature strings
        feature_str = []
        for f in self.feature_list:
            f.apply(sentence, mention, feature_str)
            # print f

        # map feature_names and label_names
        feature_ids = set()
        label_ids = set()
        for s in feature_str:
            if s in self.feature_mapping:
                feature_ids.add(self.feature_mapping[s][0])
                self.feature_mapping[s][1] += 1  # add frequency
            elif self.is_train:
                feature_ids.add(self.feature_count)
                self.feature_mapping[s] = [self.feature_count, 1]
                self.feature_count += 1
        for l in mention.labels:
            if l in self.label_mapping:
                label_ids.add(self.label_mapping[l][0])
                self.label_mapping[l][1] += 1  # add frequency
            elif self.is_train:
                label_ids.add(self.label_count)
                self.label_mapping[l] = [self.label_count, 1]
                self.label_count += 1

        return feature_ids, label_ids


def pipeline(json_file, brown_file, outdir):
    reader = MentionReader(json_file)
    ner_feature = NERFeature(is_train=True, brown_file=brown_file)
    count = 0
    gx = open(outdir+'/train_x.txt', 'w')
    gy = open(outdir+'/train_y.txt', 'w')
    f = open(outdir+'/feature.map', 'w')
    t = open(outdir+'/type.txt', 'w')
    print 'start train feature generation'
    mention_count = 0
    while reader.has_next():
        if count%10000 == 0:
            print count
        sentence = reader.next()
        for mention in sentence.mentions:
            try:
                m_id = '%s_%d_%d_%d'%(sentence.fileid, sentence.senid, mention.start, mention.end)
                feature_ids, label_ids = ner_feature.extract(sentence, mention)
                gx.write(m_id+'\t'+','.join([str(x) for x in feature_ids])+'\n')
                gy.write(m_id+'\t'+','.join([str(x) for x in label_ids])+'\n')
                mention_count += 1
                count += 1
            except Exception as e:
                print e.message, e.args
                print sentence.fileid, sentence.senid, len(sentence.tokens)
                print mention
    print 'mention :%d'%mention_count
    print 'feature :%d'%len(ner_feature.feature_mapping)
    print 'label :%d'%len(ner_feature.label_mapping)
    write_map(ner_feature.feature_mapping, f)
    write_map(ner_feature.label_mapping, t)
    reader.close()
    gx.close()
    gy.close()
    f.close()
    t.close()


def pipeline_test(json_file, brown_file, featurefile, labelfile, outdir):
    #  load feature mapping and label mapping
    feature_map = load_map(featurefile)
    label_map = load_map(labelfile)

    reader = MentionReader(json_file)
    ner_feature = NERFeature(is_train=False, brown_file=brown_file, feature_mapping=feature_map, label_mapping=label_map)
    count = 0
    gx = open(outdir+'/test_x.txt', 'w')
    gy = open(outdir+'/test_y.txt', 'w')

    print 'start test feature generation'
    while reader.has_next():
            
        if count%1000 == 0:
            print count
        sentence = reader.next()
        for mention in sentence.mentions:
            try:
                m_id = '%s_%d_%d_%d'%(sentence.fileid, sentence.senid, mention.start, mention.end)
                feature_ids, label_ids = ner_feature.extract(sentence, mention)
                if len(label_ids)>0:
                    gx.write(m_id+'\t'+','.join([str(x) for x in feature_ids])+'\n')
                    gy.write(m_id+'\t'+','.join([str(x) for x in label_ids])+'\n')
                    count += 1
            except Exception as e:
                print e.message, e.args
                print sentence.fileid, sentence.senid
                print sentence
                continue
    print count
    reader.close()
    gx.close()
    gy.close()


def load_map(input):
    f = open(input)
    mapping = {}
    for line in f:
        seg = line.strip('\r\n').split('\t')
        mapping[seg[0]] = [int(seg[1]), 0]
    f.close()
    return mapping


def write_map(mapping, output):
    sorted_map = sorted(mapping.items(),cmp=lambda x,y:x[1][0]-y[1][0])
    for tup in sorted_map:
        output.write(tup[0]+'\t'+str(tup[1][0])+'\t'+str(tup[1][1])+'\n')


def filter(featurefile, trainfile, featureout,trainout):
    f = open(featurefile)
    featuremap = {}
    old2new = {}
    count = 0
    for line in f:
        seg = line.strip('\r\n').split('\t')
        frequency = int(seg[2])
        if frequency>=2:
            featuremap[seg[0]] = (count,seg[2])
            old2new[seg[1]] = count
            count+=1
    print 'Feature after filter: %d'%count
    f.close()
    g = open(featureout,'w')
    write_map2(featuremap, g)
    g.close()

    # scan the training set and filter features
    f = open(trainfile)
    g = open(trainout,'w')
    for line in f:
        seg = line.strip('\r\n').split('\t')
        # features = line.strip('\r\n').split(',')
        features = seg[1].split(',')
        newfeatures = set()
        for feature in features:
            if feature in old2new:
                newfeatures.add(old2new[feature])
        g.write(seg[0]+'\t'+','.join([str(x) for x in newfeatures])+'\n')
        # g.write(','.join([str(x) for x in newfeatures])+'\n')

    f.close()
    g.close()


def write_map2(mapping, output):
    sorted_map = sorted(mapping.items(),cmp=lambda x,y:x[1][0]-y[1][0])
    for tup in sorted_map:
        output.write(tup[0]+'\t'+str(tup[1][0])+'\n')

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print 'Usage:ner_feature.py -TRAIN_JSON -TEST_JSON -BROWN_FILE -OUTDIR'
        exit(1)
    train_json = sys.argv[1]
    test_json = sys.argv[2]
    brown_file = sys.argv[3]
    outdir = sys.argv[4]
    pipeline(train_json, brown_file, outdir)
    filter(featurefile=outdir+'/feature.map', trainfile=outdir+'/train_x.txt', featureout=outdir+'/feature.txt',trainout=outdir+'/train_x_new.txt')
    pipeline_test(test_json, brown_file, outdir+'/feature.txt',outdir+'/type.txt', outdir)
