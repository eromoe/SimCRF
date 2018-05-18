# -*- coding: utf-8 -*-
# @Author: mithril
# @Date:   2017-04-11 10:48:05
# @Last Modified by:   mithril
# @Last Modified time: 2018-04-18 15:19:05

from __future__ import unicode_literals, print_function, absolute_import

import os
import codecs
import pickle
import unicodedata
from six import string_types
import sklearn_crfsuite
import numpy as np

from .features import tokens2offsets, CrfTransformer


class SimCRF(object):

    def __init__(self, crf_model=None, crf_model_path=None, transform_window=2, tokenizer=None, max_iterations=50, verbose=False, preiob=True):
        '''

        preiob: IOB mark can be tag head or tail, depend to trainning data
        '''

        self.crf_model = crf_model
        if not crf_model and crf_model_path:
            with open(crf_model_path, 'rb') as f:
                self.crf_model = pickle.load(f)

        self.transformer = CrfTransformer(window=transform_window, tokenizer=tokenizer)
        self.verbose= verbose
        self.max_iterations = max_iterations
        self.preiob = preiob

    def fit(self, X_train, y_train, X_test=None, y_test=None, verbose=False):
        crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
            verbose=self.verbose,
        )
        self.crf_model = crf_model.fit(X_train, y_train, X_test, y_test)

        return crf_model

    def transform(self, *args, **kwargs):
        return self.transformer.transform(*args, **kwargs)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def save_crfsute_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.crf_model, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load_crfsute_model(cls, path, **kwargs):
        with open(path, 'rb') as f:
            crf_model = pickle.load(f)
            return cls(crf_model, **kwargs)

    def pretty_entities(self, tokens, iob_tags, output='plain'):
        '''
        tokens : list of  (string, tag)
        '''

        assert output in ('token', 'plain', 'offset')

        if output == 'offset':
            offsets = tokens2offsets(tokens)

        entities = []
        start = None
        curtag = None

        for cursor, tag in enumerate(iob_tags):
            if tag.startswith('B') if self.preiob else tag.endswith('B'):
                start = cursor
                curtag = tag.split('-')[1 if self.preiob else 0 ]
            elif start and tag == 'O':
                # entities.append((start, cursor))
                if output == 'plain':
                    entities.append((curtag, ''.join(tokens[start:cursor])))
                elif output == 'token':
                    entities.append(tokens[start:cursor])
                elif output == 'offset':
                    tk_offset = (offsets[start], offsets[cursor])
                    # print(tk_offset)
                    # print(texts[idx][tk_offset[0]:tk_offset[1]])
                    # print(''.join(i[0] for i in X[idx][start:cursor]))
                    # assert texts[idx][tk_offset[0]:tk_offset[1]] == ''.join(i[0] for i in X[idx][start:cursor])
                    entities.append((curtag, tk_offset))

                start = None
                curtag = None

        return entities

    def extract_taggedtokens(self, tagedtokens, output='plain'):
        '''
            tagedtokens: list of (token, tag)
        '''
        tokens = [token for token, tag in tagedtokens]
        X_features = self.transformer.transform_one(tagedtokens)
        y = self.crf_model.predict_single(X_features)
        return self.pretty_entities(tokens, y, output=output)

    def extract(self, sent=None, tokens=None, tags=None, output='plain'):
        '''
            sent: string
            tokens: list of token
            tags: list of tag
        '''

        assert self.transformer.tokenizer is not None

        if sent:
            tokens = self.transformer.tokenizer(sent)
            X_features = self.transformer.transform_one(tokens)
            if not isinstance(tokens[0], string_types):
                # tokens is tagged, extract need only token list
                tokens = [t[0] for t in tokens]

        elif tokens:
            if tags:
                X_features = self.transformer.transform_one(list(zip(tokens, tags)))
            else:
                X_features = self.transformer.transform_one(tokens)
        else:
            raise Exception('Invalid input! Must have sent or tokens')

        y = self.crf_model.predict_single(X_features)

        return self.pretty_entities(tokens, y, output=output)


