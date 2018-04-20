# -*- coding: utf-8 -*-
# @Author: mithril

from __future__ import unicode_literals, print_function, absolute_import

import json
import codecs
import jieba.posseg as pseg


def is_string(obj):
    try:
        return isinstance(obj, basestring)
    except NameError:
        return isinstance(obj, str)

def read_iob_file(input_path):
    if input_path.endswith('.txt'):
        with codecs.open(input_path, 'r', 'utf-8') as f:
            sents = f.read().split('\n\n')
            sents = list(map(lambda x: [i.split() for i in x]  , ( s.split('\n') for s in sents)))
    elif input_path.endswith('.json'):
        with codecs.open(input_path, 'r', 'utf-8') as f:
            sents = json.loads(f.read())
    else:
        raise NotImplementedError()

    return sents

def pos_range_cut(sentence, HMM=True):
    start = 0
    for w in pseg.cut(sentence, HMM=HMM):
        yield w, (start, start+len(w.word))
        start += len(w.word)

def pos_offset_cut(sentence, HMM=True):
    start = 0
    for w in pseg.cut(sentence, HMM=HMM):
        yield w, start
        start += len(w.word)

def pos_cut(text):
    return [tuple(pair) for pair in pseg.cut(text)]