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
import sklearn_crfsuite
import numpy as np

from .features import tokens2offsets, CrfTrasformer


class SimCRF(object):

    def __init__(self, crf_model=None, crf_model_path=None, transform_window=2, tokenizer=None):
        self.crf_model = crf_model
        if not crf_model and crf_model_path:
            with open(crf_model_path, 'rb') as f:
                self.crf_model = pickle.load(f)

        self.transformer = CrfTrasformer(window=transform_window, tokenizer=tokenizer)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.crf_model = crf_model.fit(X_train, y_train, X_test, y_test)

        return crf_model

    def transform(self, *args, **kwargs):
        return self.transformer.transform(*args, **kwargs)

    def save(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.crf_model, f)

    def pretty_entities(self, tokens, iob_tags, output='plain'):
        '''
        items : list of  (string, tag)
        '''

        assert output in ('token', 'plain', 'offset')

        if output == 'offset':
            offsets = tokens2offsets(tokens)

        entities = []
        start = None
        curtag = None

        for cursor, tag in enumerate(iob_tags):
            if tag.endswith('B'):
                start = cursor
                curtag = tag.split('-')[0]
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
        if sent:
            X_features = self.transformer.transform_one(tokens, tokenize=True)
        elif tokens:
            if tags:
                X_features = self.transformer.transform_one(list(zip(tokens, tags)))
            else:
                X_features = self.transformer.transform_one(tokens)
        else:
            raise Exception('Invalid input! Must have sent or tokens')

        y = self.crf_model.predict_single(X_features)

        return self.pretty_entities(tokens, y, output=output)


