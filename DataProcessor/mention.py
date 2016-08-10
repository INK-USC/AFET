

class Mention(object):
    """
    Wrap a mention. The entity name of the mention is sentence.tokens[start:end].
    Attributes
    ==========
    start : int
        The start index of the mention.
    end : int
        The end index of the mention.(not included)
    labels : list
        The labels.
    """
    def __init__(self, start, end, labels):
        self.start = start
        self.end = end
        self.labels = labels

    def __str__(self):
        result = 'start: %d, end: %d\n' % (self.start, self.end)
        for label in self.labels:
            result += label + ','
        return result


class Sentence(object):
    """
    Wrap a sentence.
    Attributes
    ==========
    fileid : string
        The file id.
    senid : string
        The sentence id.
    tokens : list
        The token list of this sentence.
    """
    def __init__(self, fileid, senid, tokens):
        self.fileid = fileid
        self.senid = senid
        self.tokens = tokens
        self.mentions = []
        self.pos = []
        self.dep = []

    def __str__(self):
        result = 'fileid: %s, senid: %s\n'%(self.fileid, self.senid)
        for token in self.tokens:
            result += token + ' '
        result += '\n'
        for m in self.mentions:
            result += m.__str__() + '\n'
        return result

    def add_mention(self, mention):
        assert isinstance(mention, Mention)
        self.mentions.append(mention)

    def size(self):
        return min(len(self.tokens),len(self.pos))


