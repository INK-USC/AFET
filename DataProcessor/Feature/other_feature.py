
import re
from abstract_feature import AbstractFeature
from token_feature import HeadFeature

class PosFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        for i in xrange(mention.start, mention.end):
            features.append('POS_%s' % sentence.pos[i])


class LengthFeature(AbstractFeature):

    def apply(self, sentence, mention, features):
        length = mention.end - mention.start
        if length <= 5:
            features.append('LENGTH_%d' % length)
        else:
            features.append('LENGTH_>5')


class WordShapeFeature(AbstractFeature):
    def get_word_shape(self, token):
        result = re.sub('[a-z]+', 'a', token)
        result = re.sub('[A-Z]+', 'A', result)
        result = re.sub('[0-9]+', '0', result)
        result = re.sub(ur"\p{P}+", '.', result)
        return result

    def apply(self, sentence, mention, features):
        for i in xrange(mention.start, mention.end):
            features.append('SHAPE_%s' % self.get_word_shape(sentence.tokens[i]))


class CharacterFeature(AbstractFeature):
    def apply(self, sentence, mention, features):
        head_index = HeadFeature.get_head(sentence, mention)
        head = sentence.tokens[head_index]
        if len(head) >= 3:
            for i in xrange(0, len(head)-2):
                features.append('CHAR_%s' % head[i:(i + 3)])
            features.append('CHAR_:%s' % head[:2])
            features.append('CHAR_%s:' % head[(len(head)-2):])
