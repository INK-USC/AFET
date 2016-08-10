from __future__ import division

import sys
import re
from collections import defaultdict

def map_type(entityfile, mapfile, output):
    type_map = {}
    f = open(mapfile)
    for line in f:
        seg = line.strip('\r\n').split('\t')
        type_map[seg[0]] = seg[1]
    f.close()
    print type_map
    type_entities = defaultdict(set)
    f = open(entityfile)
    for line in f:
        seg = line.strip('\r\n').split('\t')
        freebase_label = re.sub('\.','/',seg[0][27:-1])
     #   print freebase_label
        if freebase_label in type_map:
            type_entities[type_map[freebase_label]].update(seg[1].split(';'))

    f.close()
    print len(type_entities)
    g = open(output,'w')
    for key in type_entities:
        g.write(key+'\t'+';'.join(type_entities[key])+'\n')
    g.close()


def share_entity(entityfile, labelmap, output):
    # load label mapping
    f = open(labelmap)
    label_map = {}
    for line in f:
        seg = line.strip('\r\n').split('\t')
        label_map[seg[0]] = int(seg[1])
    f.close()

    # load type entities
    f = open(entityfile)
    type_entities = {}
    for line in f:
        seg = line.strip('\r\n').split('\t')
        if seg[0] in label_map:
            type_entities[label_map[seg[0]]] = set(seg[1].split(';'))

    size = len(label_map)
    g = open(output,'w')
    for i in xrange(0,size):
        for j in xrange(0,size):
            if i!=j:
                if i in type_entities and j in type_entities:
                    Ei = type_entities[i]
                    Ej = type_entities[j]
                    scoreij = len(Ei & Ej) / len(Ei)
                    scoreji = len(Ei & Ej) / len(Ej)
                    if scoreij+scoreji > 0.000001:
                        g.write(str(i)+'\t'+str(j)+'\t'+str((scoreij+scoreji)/2)+'\n')
    g.close()


 # Run TypeEntity instance
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print 'Usage:type_type_kb.py -INDIR'
        exit(1)
    # map freebase to target types
    type_entities = indir + '/type_entities.txt'
    label_map = indir + '/type.txt'
    output = indir + '/type_type_kb.txt'
    share_entity(type_entities, label_map, output)






