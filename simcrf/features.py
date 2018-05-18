# -*- coding: utf-8 -*-
# @Author: mithril

from __future__ import unicode_literals, print_function, absolute_import

from six import string_types


def sent2labels(sent):
    return (label for token, postag, label in sent)

def sent2tokens(sent):
    return (token for token, postag, label in sent)

def tokens2offsets(tokens):
    offsets = []
    start = 0
    for token in tokens:
        offsets.append(start)
        start += len(token)
    return offsets

class CrfTransformer(object):
    def __init__(self, window=2, tokenizer=None):
        self.window = window
        self.tokenizer = tokenizer

    def _transform_one(self, tokens):
        '''
        tokens: list of token(str) or list of (token, tag)
        '''
        zipped = not isinstance(tokens[0], string_types)
        return self.taggedtokens2features(tokens) if zipped else self.tokens2features(tokens)

    def transform_one(self, sent, tokenize=False):
        '''
        sent: 1. string 
              2.list of token(str)
              3.list of (token, tag)
        '''
        if tokenize:
            sent = self.tokenizer(sent)

        return self._transform_one(sent)


    def transform(self, sents, tokenize=False):
        '''
        sents : list of words, words can be list of token(str) or list of (token, tag)
        '''
        if tokenize:
            return (self._transform_one(self.tokenizer(s)) for s in sents)
        else:
            return (self._transform_one(s) for s in sents)

    

    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]
        window = self.window + 1

        features = {
            'bias': 1.0,
            'word': word,
            'word[-1:]': word[-1:],
            'word[-2:]': word[-2:],
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            # 'postag[:2]': postag[:2],
        }

        if i <= 0:
            features['BOS'] = True
        else:
            for left in range(1, window):
                if i > left-1:
                    features.update({
                        '-%s:word' % left: sent[i-left][0],
                        '-%s:postag' % left: sent[i-left][1],
                    })

        if i >= len(sent) :
            features['EOS'] = True
        else:
            for right in range(1, window):
                if i < len(sent)- right:
                    features.update({
                        '+%s:word' % right: sent[i+right][0],
                        '+%s:postag' % right: sent[i+right][1],
                    })

        return features

    def token2features(self, tokens, i):
        word = tokens[i]
        window = self.window + 1

        features = {
            'bias': 1.0,
            'word': word,
            'word[-1:]': word[-1:],
            'word[-2:]': word[-2:],
            'word.isdigit()': word.isdigit(),
        }

        if i <= 0:
            features['BOS'] = True
        else:
            for left in range(1, window):
                if i > left-1:
                    features.update({
                        '-%s:word' % left: tokens[i-left],
                    })

        if i >= len(tokens) :
            features['EOS'] = True
        else:
            for right in range(1, window):
                if i < len(tokens)- right:
                    features.update({
                        '+%s:word' % right: tokens[i+right],
                    })

        return features

    def tag2features(self, tags, i):
        postag = tags[i]
        window = self.window + 1

        features = {
            'postag': postag,
            # 'postag[:2]': postag[:2],
        }

        if i <= 0:
            features['BOS'] = True
        else:
            for left in range(1, window):
                if i > left-1:
                    features.update({
                        '-%s:postag' % left: tags[i-left],
                    })

        if i >= len(tags) :
            features['EOS'] = True
        else:
            for right in range(1, window):
                if i < len(tags) - right:
                    features.update({
                        '+%s:postag' % right: tags[i+right],
                    })

        return features

    def taggedtokens2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def tokens2features(self, sent):
        return [self.token2features(sent, i) for i in range(len(sent))]

