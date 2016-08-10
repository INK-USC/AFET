
import sys
import os
from collections import defaultdict
from math import sqrt
import operator


def sim_func(v1, v2, _MODE):
    val = 0.0
    if _MODE == 'dot':
        ### dot product:
        val = sum( [v1[i]*v2[i] for i in range(len(v1))] )
    elif _MODE == 'cosine':
        ### cosine sim:
        norm1 = sqrt(sum( [v1[i]*v1[i] for i in range(len(v1))] ))
        norm2 = sqrt(sum( [v2[i]*v2[i] for i in range(len(v1))] ))
        val = sum( [v1[i]*v2[i]/norm1/norm2 for i in range(len(v1))] )
    return val

# Manual type hierarchy
class TypeHierarchy:
    def __init__(self, file_name, number_of_types):
        self.type_hierarchy = {} # type -> [parent type]
        self.subtype_mapping = defaultdict(list) # type -> [subtype]
        self._root = set() # root types (on 1-level)
        with open(file_name) as f:
            for line in f:
                t = line.strip('\r\n').split('\t')
                self.type_hierarchy[int(t[0])] = int(t[1])
                self.subtype_mapping[int(t[1])].append(int(t[0]))
                self._root.add(int(t[0]))
        self._root = list(set(range(0, number_of_types)).difference(self._root))
        # print self._root
        # print self.subtype_mapping

    def get_type_hierarchy(self):
        return self.type_hierarchy

# Embedding of different nodes
class Embedding:
    def __init__(self, file_name):
        self._embs = []
        self._node_size = 0
        self._vector_size = 0
        # load file to embedding array
        with open(file_name) as f:
            seg = f.readline().split(' ')
            self._node_size = int(seg[0])
            self._vector_size = int(seg[1])
            # self._embs = [np.zeros(self._vector_size) for i in range(self._node_size)]
            self._embs = [[] for i in range(self._node_size)]
            for line in f:
                seg = line.strip().split('\t')
                idx = int(seg[1])
                _emb = seg[2].split(' ')
                # self._embs[idx] = np.array([float(x) for x in _emb])
                self._embs[idx] = [float(x) for x in _emb]
        # print 'emb:', self._node_size, self._vector_size


    def get_embedding(self, index):
        return self._embs[index]

class Network:
    def __init__(self, file_name):
        self._network = defaultdict(list)
        # load file to network dictionary
        cnt = 0
        with open(file_name) as f:
            for line in f:
                seg = line.strip('\r\n').split('\t')
                self._network[int(seg[0])].append(int(seg[1]))
                cnt += 1
        # print 'edges:', cnt

    # return list of features
    def get_neighbors(self, idx):
        return self._network[idx]


# Predict types from feature embeddings
class Predicter_useFeatureEmb:
    def __init__(self, embs_feature, embs_type, network_mention_feature, supertypefile, sim_func):
        self._embs_feature = Embedding(embs_feature)
        self._embs_type = Embedding(embs_type)
        assert self._embs_feature._vector_size == self._embs_type._vector_size
        self._network_mention_feature = Network(network_mention_feature)
        self._type_hierarchy = TypeHierarchy(supertypefile, self._embs_type._node_size)
        self._sim_func = sim_func

    # get embedding vector for a mention
    def get_mention_embedding(self, mention_id):
        # from _network_mention_feature & _emb_feature
        feature_list = self._network_mention_feature.get_neighbors(mention_id)
        # _emb_mention = np.zeros(self._embs_feature._vector_size)
        _emb_mention = [0.0 for i in range(self._embs_feature._vector_size)]
        # for feature_id in feature_list:
        #     _emb_mention += self._embs_feature.get_embedding(feature_id) / float(len(feature_list))
        for feature_id in feature_list:
            for i in range(self._embs_feature._vector_size):
                _emb_mention[i] += self._embs_feature.get_embedding(feature_id)[i] / float(len(feature_list))
        return _emb_mention

    # predict types given a mention embedding
    def predict_types_for_mention_maximum(self, mention_id, _threshold):
        _type_size = self._embs_type._node_size
        _emb_mention = self.get_mention_embedding(mention_id)

        parent_mapping = self._type_hierarchy.get_type_hierarchy()

        labels = []
        scores = []
        # calculate scores and find maximum score
        max_index = -1
        max_score = -sys.maxint
        for i in xrange(_type_size):
            _emb_type = self._embs_type.get_embedding(i)
            score = sim_func(_emb_mention, _emb_type, self._sim_func)
            scores.append(score)
            if max_score < score:
                    max_index = i
                    max_score = score
        labels.append(max_index)
        
        # Add parent of max_index if any
        temp = max_index
        while temp in parent_mapping:
            labels.append(parent_mapping[temp])
            temp = parent_mapping[temp]

        ### add child of max_index if meeting threshold
        temp = max_index
        while temp != -1:
            max_sub_index = -1
            max_sub_score = -sys.maxint
            for child in parent_mapping:
                # check the maximum subtype
                if parent_mapping[child] == temp:
                    if max_sub_score < scores[child]:
                        max_sub_index = child
                        max_sub_score = scores[child]
            if max_sub_index != -1 and max_sub_score > _threshold:
                labels.append(max_sub_index)
            temp = max_sub_index
        
        return labels

    # predict types given a mention embedding. Method 2, top-down
    def predict_types_for_mention_topDown(self, mention_id, _threshold):
        _type_size = self._embs_type._node_size
        _emb_mention = self.get_mention_embedding(mention_id)

        labels = []
        # calculate scores and find maximum score
        root = self._type_hierarchy._root
        while root is not None:
            # find maximum index in this level 
            max_index = -1
            max_score = -sys.maxint
            for i in root:
                _emb_type = self._embs_type.get_embedding(i)
                score = sim_func(_emb_mention, _emb_type, self._sim_func)
                if max_score < score:
                    max_index = i
                    max_score = score
            
            # Add if it's a root type, or it meets threshold
            if max_index in self._type_hierarchy._root or max_score > _threshold:
                labels.append(max_index)
            else:
                break

            # Check subtypes if any
            if max_index in self._type_hierarchy.subtype_mapping:
                root = self._type_hierarchy.subtype_mapping[max_index]
            else:
                break

        return labels

    # predict types given a mention embedding. Method 3, flat
    def predict_types_for_mention_topk(self, mention_id, _threshold):
        K = 2 # max depth of type hierarchy
        _type_size = self._embs_type._node_size
        root_set = set(self._type_hierarchy._root)
        parent_mapping = self._type_hierarchy.get_type_hierarchy()
        _emb_mention = self.get_mention_embedding(mention_id)
        
        scores = defaultdict(float) # label -> score
        for i in xrange(_type_size):
            _emb_type = self._embs_type.get_embedding(i)
            score = sim_func(_emb_mention, _emb_type, self._sim_func)
            if score >= _threshold:
                scores[i] = score
        
        if len(scores) == 0:
            return [-1]

        # Get top-K labels
        topk = set()
        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(min(K, len(scores))):
            topk.add(sorted_scores[i][0])

        ### resolve conflicts
        labels = []
        overlap = topk & root_set
        # has no root type: get parent for the best
        if len(overlap) == 0:
            max_t = -sys.maxint
            max_score = -sys.maxint
            for i in topk:
                if max_score < scores[i]:
                    max_score = scores[i]
                    max_t = i
            labels.append(max_t)
            labels.append(parent_mapping[max_t])

        # has 1 root type: add subtypes if any
        elif len(overlap) == 1:
            tmp = list(overlap)
            labels.append(tmp[0])
            topk.discard(tmp[0])
            for t in topk: # check subtypes
                if t in parent_mapping and parent_mapping[t] == labels[0]:
                    labels.append(t)

        # has more than 1 root types: get the best, add subtypes if any
        elif len(overlap) > 1:
            max_t = -sys.maxint
            max_score = -sys.maxint
            for i in overlap:
                if max_score < scores[i]:
                    max_score = scores[i]
                    max_t = i
            labels.append(max_t)
            topk.discard(max_t)
            for t in topk: # check subtypes
                if t in parent_mapping and parent_mapping[t] == labels[0]:
                    labels.append(t)

        return labels


# Predict types for mentions in test data
class Predicter_useMentionEmb:
    def __init__(self, embs_mention, embs_type, supertypefile, sim_func):
        self._embs_mention = Embedding(embs_mention)
        self._embs_type = Embedding(embs_type)
        assert self._embs_mention._vector_size == self._embs_type._vector_size
        self._type_hierarchy = TypeHierarchy(supertypefile, self._embs_type._node_size)
        self._sim_func = sim_func

    # get embedding vector for a mention
    def get_mention_embedding(self, mention_id):
        if mention_id < 0 or mention_id > self._embs_mention._node_size:
            print 'mention NOT in embs_mention!'
            exit(0)
        return self._embs_mention.get_embedding(mention_id)


    # predict types given a mention embedding
    def predict_types_for_mention_maximum(self, mention_id, _threshold):
        _type_size = self._embs_type._node_size
        _emb_mention = self.get_mention_embedding(mention_id)

        parent_mapping = self._type_hierarchy.get_type_hierarchy()

        labels = []
        scores = []
        # calculate scores and find maximum score
        max_index = -1
        max_score = -sys.maxint
        for i in xrange(_type_size):
            _emb_type = self._embs_type.get_embedding(i)
            score = sim_func(_emb_mention, _emb_type, self._sim_func)
            scores.append(score)
            if max_score < score:
                    max_index = i
                    max_score = score
        labels.append(max_index)
        
        # Add parent of max_index if any
        temp = max_index
        while temp in parent_mapping:
            labels.append(parent_mapping[temp])
            temp = parent_mapping[temp]

        ### add child of max_index if meeting threshold
        temp = max_index
        while temp != -1:
            max_sub_index = -1
            max_sub_score = -sys.maxint
            for child in parent_mapping:
                # check the maximum subtype
                if parent_mapping[child] == temp:
                    if max_sub_score < scores[child]:
                        max_sub_index = child
                        max_sub_score = scores[child]
            if max_sub_index != -1 and max_sub_score > _threshold:
                labels.append(max_sub_index)
            temp = max_sub_index
        
        return labels

    # predict types given a mention embedding. Method 2, top-down
    def predict_types_for_mention_topDown(self, mention_id, _threshold):
        _type_size = self._embs_type._node_size
        _emb_mention = self.get_mention_embedding(mention_id)

        labels = []
        # calculate scores and find maximum score
        root = self._type_hierarchy._root
        while root is not None:
            # find maximum index in this level 
            max_index = -1
            max_score = -sys.maxint
            for i in root:
                _emb_type = self._embs_type.get_embedding(i)
                score = sim_func(_emb_mention, _emb_type, self._sim_func)
                if max_score < score:
                    max_index = i
                    max_score = score
            
            # Add if it's a root type, or it meets threshold
            if max_index in self._type_hierarchy._root or max_score > _threshold:
                labels.append(max_index)
            else:
                break

            # Check subtypes if any
            if max_index in self._type_hierarchy.subtype_mapping:
                root = self._type_hierarchy.subtype_mapping[max_index]
            else:
                break

        return labels

    # predict types given a mention embedding. Method 3, flat
    def predict_types_for_mention_topk(self, mention_id, _threshold):
        K = 2 # max depth of type hierarchy
        _type_size = self._embs_type._node_size
        root_set = set(self._type_hierarchy._root)
        parent_mapping = self._type_hierarchy.get_type_hierarchy()
        _emb_mention = self.get_mention_embedding(mention_id)
        
        scores = defaultdict(float) # label -> score
        for i in xrange(_type_size):
            _emb_type = self._embs_type.get_embedding(i)
            score = sim_func(_emb_mention, _emb_type, self._sim_func)
            if score >= _threshold:
                scores[i] = score
        
        if len(scores) == 0:
            return [-1]

        # Get top-K labels
        topk = set()
        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(min(K, len(scores))):
            topk.add(sorted_scores[i][0])

        ### resolve conflicts
        labels = []
        overlap = topk & root_set
        # has no root type: get parent for the best
        if len(overlap) == 0:
            max_t = -sys.maxint
            max_score = -sys.maxint
            for i in topk:
                if max_score < scores[i]:
                    max_score = scores[i]
                    max_t = i
            labels.append(max_t)
            labels.append(parent_mapping[max_t])

        # has 1 root type: add subtypes if any
        elif len(overlap) == 1:
            tmp = list(overlap)
            labels.append(tmp[0])
            topk.discard(tmp[0])
            for t in topk: # check subtypes
                if t in parent_mapping and parent_mapping[t] == labels[0]:
                    labels.append(t)

        # has more than 1 root types: get the best, add subtypes if any
        elif len(overlap) > 1:
            max_t = -sys.maxint
            max_score = -sys.maxint
            for i in overlap:
                if max_score < scores[i]:
                    max_score = scores[i]
                    max_t = i
            labels.append(max_t)
            topk.discard(max_t)
            for t in topk: # check subtypes
                if t in parent_mapping and parent_mapping[t] == labels[0]:
                    labels.append(t)

        # if len(labels) == 1: return [-1]
        return labels


def predict(indir, outdir, _method, _emb_mode, _predict_mode, _sim_func, _threshold):
    # predict the testing mentions' labels
    if _emb_mode == 'bipartite':
        predicter = Predicter_useFeatureEmb(\
            embs_feature=os.path.join(outdir + '/emb_' + _method + '_bipartite_feature.txt'), \
            embs_type=os.path.join(outdir + '/emb_' + _method + '_bipartite_type.txt'), \
            network_mention_feature=os.path.join(indir + '/mention_feature_test.txt'), \
            supertypefile=os.path.join(indir + '/supertype.txt'), \
            sim_func=_sim_func)

    elif _emb_mode == 'hete_mention':
        predicter = Predicter_useMentionEmb(\
            embs_mention=os.path.join(outdir + '/emb_' + _method + '_mention.txt'), \
            embs_type=os.path.join(outdir + '/emb_' + _method + '_type_test.txt'), \
            supertypefile=os.path.join(indir + '/supertype.txt'), \
            sim_func=_sim_func)

    elif _emb_mode == 'hete_feature':        
        predicter = Predicter_useFeatureEmb(\
            embs_feature=os.path.join(outdir + '/emb_' + _method + '_feature.txt'), \
            embs_type=os.path.join(outdir + '/emb_' + _method + '_type_test.txt'), \
            network_mention_feature=os.path.join(indir + '/mention_feature.txt'), \
            supertypefile=os.path.join(indir + '/supertype.txt'), \
            sim_func=_sim_func)

    else:
        print 'wrong parameter!'
        exit(-1)

    with open(os.path.join(indir + '/mention_type_test.txt')) as f,\
         open(os.path.join(outdir + '/mention_type_' + _method + '_' + _emb_mode + '.txt'), 'w') as g:
        cnt = 0
        mentions_tested = set()
        for line in f:
            seg = line.strip('\r\n').split('\t')
            mention_id = int(seg[0])
            if mention_id not in mentions_tested:
                mentions_tested.add(mention_id)
                if _predict_mode == 'maximum':
                    labels = predicter.predict_types_for_mention_maximum(mention_id, _threshold)
                elif _predict_mode == 'topdown':
                    labels = predicter.predict_types_for_mention_topDown(mention_id, _threshold)
                elif _predict_mode == 'topk':
                    labels = predicter.predict_types_for_mention_topk(mention_id, _threshold)
                for l in labels:
                    g.write(str(mention_id)+'\t'+str(l)+'\t'+'1\n')
                cnt += 1
        f.close()
        g.close()
    print cnt, 'mentions predicted.'


def load_mentionids(filename):
    """
    Load mention id as a set.
    e.g.[[0,1,2],[1,2]]
    """
    with open(filename) as f:
        indexes = set()
        for line in f:
            seg = line.strip('\r\n').split('\t')
            indexes.add(int(seg[0]))
        return indexes

def load_candidates(filename, indexes):
    """
    Load data as a dict of list.
    e.g.{0:[0,1,2],1:[1,2]}
    """
    with open(filename) as f:
        data = defaultdict(list)
        for line in f:
            seg = line.strip('\r\n').split('\t')
            index = int(seg[0])
            if index in indexes:
                data[index].append(int(seg[1]))
        return data


if __name__ == "__main__":
    
    if len(sys.argv) != 7:
        print 'Usage: emb_prediction.py -DATA(FIGER) -METHOD(pte) -EMB_MODE(hete_feature) -PREDICT_MODE(topdown) -SIM(cosine/dot) -THRESHOLD'
        exit(-1)

    # do prediction here
    _data = sys.argv[1]
    _method = sys.argv[2]
    _emb_mode = sys.argv[3] # reduce_label_noise / typing
    _predict_mode = sys.argv[4] # topdown / maximum manner
    _sim_func = sys.argv[5]
    _threshold = float(sys.argv[6])

    indir = 'Intermediate/' + _data
    outdir = 'Results/' + _data
    predict(indir, outdir, _method, _emb_mode, _predict_mode, _sim_func, _threshold)



