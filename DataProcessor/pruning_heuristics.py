
import os
import operator
import sys
from collections import defaultdict
reload(sys)
sys.setdefaultencoding('utf8')

class TypeHierarchy:
    def __init__(self, file_name):
        self._type_hierarchy = {}
        with open(file_name) as f:
            for line in f:
                t = line.strip('\r\n').split('\t')
                # t[0] is a subtype of t[1]
                self._type_hierarchy[int(t[0])] = (int(t[1]))
        print self._type_hierarchy

    def get_type_path(self, label):
        if label in self._type_hierarchy:  # label has super type
            path = [label]
            while label in self._type_hierarchy:
                path.append(self._type_hierarchy[label])
                label = self._type_hierarchy[label]
            path.reverse()
            return path
        else:  # label is the root type
            return [label]


class TypeDistribution:
    def __init__(self, file_name):
        self.type_distribution = {}
        with open(file_name) as f:
            for line in f:
                seg = line.strip('\r\n').split('\t')
                labels = [int(x) for x in seg[1].split(',')]
                self.type_distribution[seg[0]] = labels

    def get_frequent_labels(self, fileid):
        if fileid in self.type_distribution:
            return self.type_distribution[fileid]
        return None


class PruneStrategy:
    def __init__(self, strategy, hierarchy_file, distribution_file):
        self._strategy = strategy
        self._type_hierarchy = TypeHierarchy(hierarchy_file)
        self._type_distribution = TypeDistribution(distribution_file)
        self.pruner = self.no_prune
        if strategy == 'min':
            self.pruner = self.min_prune
        elif strategy == 'all':
            self.pruner = self.all_prune
        elif strategy == 'sibling':
            self.pruner = self.sibling_prune

    def no_prune(self, fileid, is_ground, labels):
        new_labels = set(labels)
        for l in labels:
            new_labels.update(self._type_hierarchy.get_type_path(l))
        return list(new_labels)

    def min_prune(self, fileid, is_ground, labels):
        labels = self.no_prune(fileid, is_ground,labels)
        frequent_labels = self._type_distribution.get_frequent_labels(fileid)
        if frequent_labels is not None:
            refined_labels = [x for x in labels if x in frequent_labels]
            if len(refined_labels) > 0:
                refined_labels = self.no_prune(fileid, is_ground,refined_labels)
                return refined_labels
        if is_ground or fileid == '':  # indicate it is a test mention, which does not have a type distribution
            return labels
        return None

    def sibling_prune(self, fileid, is_ground, labels):
        # indicate it is a test mention, which should not be pruned
        paths = []
        for l in labels:
            paths.append(self._type_hierarchy.get_type_path(l))
        # print paths
        # find the most fine-grained common parent
        new_labels = list(paths[0])
        for p in xrange(1, len(labels)):
            path1 = list(paths[p])
            i = 0
            while i < min(len(path1), len(new_labels)):
                if path1[i] != new_labels[i]:
                    break
                i += 1
            if i < min(len(path1), len(new_labels)):  # break is called
                for j in xrange(0, len(new_labels)-i):
                    new_labels.pop()
            elif len(path1) > len(new_labels):
                    new_labels = path1
            if len(new_labels) == 0:
                break
        if len(new_labels) > 0:
            new_labels = self.no_prune(fileid, is_ground, new_labels)
            return new_labels
        elif is_ground:
            new_labels = set()
            for p in xrange(0, len(labels)):
                new_labels.add(paths[p][0])
            return list(new_labels)
        return None

    def all_prune(self, fileid, is_ground, labels):
        refined_labels = self.sibling_prune(fileid, is_ground , labels)
        if refined_labels is not None:
            return self.min_prune(fileid, is_ground, refined_labels)
        return None


def prune(indir, outdir, strategy, feature_number, type_number):
    hierarchy_file = os.path.join(indir+'/supertype.txt')
    type_distribution = os.path.join((indir+ '/distribution_per_doc.txt'))
    prune_strategy = PruneStrategy(strategy=strategy, hierarchy_file=hierarchy_file, distribution_file=type_distribution)

    mids = {}
    ground_truth = set()
    count = 0
    train_y = os.path.join(indir+'/train_y.txt')
    train_x = os.path.join(indir+'/train_x_new.txt')
    test_x = os.path.join(indir+'/test_x.txt')
    test_y = os.path.join(indir+ '/test_y.txt')
    mention_file = os.path.join(outdir+ '/mention.txt')
    mention_type = os.path.join(outdir+ '/mention_type.txt')
    mention_feature = os.path.join(outdir+ '/mention_feature.txt')
    mention_type_test = os.path.join(outdir+'/mention_type_test.txt')
    mention_feature_test = os.path.join(outdir+ '/mention_feature_test.txt')
    feature_type = os.path.join(outdir+ '/feature_type.txt')
    # generate mention_type, and mention_feature for the training corpus
    with open(train_x) as fx, open(train_y) as fy, open(test_y) as ft, \
        open(mention_type,'w') as gt, open(mention_feature,'w') as gf:
        for line in ft:
            seg = line.strip('\r\n').split('\t')
            ground_truth.add(seg[0])
        # generate mention_type and mention_feature
        for line in fy:
            line2 = fx.readline()
            seg = line.strip('\r\n').split('\t')
            seg_split = seg[0].split('_')
            fileid = '_'.join(seg_split[:-3])
            labels = [int(x) for x in seg[1].split(',')]
            new_labels = prune_strategy.pruner(fileid=fileid, is_ground=(seg[0] in ground_truth), labels=labels)
            if new_labels is not None:
                seg2 = line2.strip('\r\n').split('\t')
                features = seg2[1].split(',')
                if seg[0] in mids:
                    continue
                for l in new_labels:
                    gt.write(str(count)+'\t'+str(l)+'\t1\n')
                for f in features:
                    gf.write(str(count)+'\t'+f+'\t1\n')
                mids[seg[0]] = count
                count += 1
                if count%200000==0:
                    print count
    # generate mention_type_test, and mention_feature_test for the test corpus
    print count
    print 'start test'
    with open(test_x) as fx, open(test_y) as fy,\
        open(mention_type_test,'w') as gt, open(mention_feature_test, 'w') as gf:
        # generate mention_type and mention_feature
        for line in fy:
            line2 = fx.readline()
            seg = line.strip('\r\n').split('\t')
            labels = [int(x) for x in seg[1].split(',')]
            seg2 = line2.strip('\r\n').split('\t')
            features = seg2[1].split(',')
            if seg[0] in mids:
                mid = mids[seg[0]]
            else:
                mid = count
               # print line2
                mids[seg[0]] = count
                count += 1
            for l in labels:
                gt.write(str(mid)+'\t'+str(l)+'\t1\n')
            for f in features:
                gf.write(str(mid)+'\t'+f+'\t1\n')
    print count
    print 'start mention part'
    # generate mention.txt
    with open(mention_file,'w') as m:
        sorted_mentions = sorted(mids.items(), key=operator.itemgetter(1))
        for tup in sorted_mentions:
            m.write(tup[0]+'\t'+str(tup[1])+'\n')
    print 'start feature_type part'
    with open(mention_feature) as f1, open(mention_type) as f2,\
        open(feature_type,'w') as g:
        fm = defaultdict(set)
        tm = defaultdict(set)
        for line in f1:
            seg = line.strip('\r\n').split('\t')
            i = int(seg[0])
            j = int(seg[1])
            fm[j].add(i)
        for line in f2:
            seg = line.strip('\r\n').split('\t')
            i = int(seg[0])
            j = int(seg[1])
            tm[j].add(i)
        for i in xrange(feature_number):
            for j in xrange(type_number):
                temp = len(fm[i]&tm[j])
                if temp > 0:
                    g.write(str(i)+'\t'+str(j)+'\t'+str(temp)+'\n')
