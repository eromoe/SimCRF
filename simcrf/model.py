# -*- coding: utf-8 -*-
# @Author: mithril
# @Date:   2017-04-11 10:48:05
# @Last Modified by:   mithril
# @Last Modified time: 2017-05-02 10:48:43

from __future__ import unicode_literals, print_function, absolute_import

import os
import codecs
import pickle
import unicodedata
import sklearn_crfsuite
import numpy as np

from .features import tokens2offsets, taggedtokens2features, tokens2features


class SimCRF(object):

    def __init__(self, crf_model=None, crf_model_path=None):
        self.crf_model = crf_model
        if not crf_model and crf_model_path:
            with open(crf_model_path, 'rb') as f:
                self.crf_model = pickle.load(f)

    def transform_one(self, tokens):
        '''
        sent: list of token(str) or list of (token, tag)
        '''
        zipped = not isinstance(sent[0], basestring)
        return taggedtokens2features(sent) if zipped else tokens2features(tokens)

    def transform(self, sents):
        '''
        sents : list of words, words can be list of token(str) or list of (token, tag)
        '''
        return [self.transform_one(s) for s in sents]

    def fit(self, X_train, y_train):
        crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.crf_model = crf_model.fit(X_train, y_train)

        return crf_model

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

    def extract_taggedtokens(self, sent, output='plain'):
        '''
            sent: list of (token, tag)
        '''
        tokens = [token for token, tag in sent]
        X_features = self.transform_one(sent)
        y = self.crf_model.predict_single(X_features)

        return self.pretty_entities(tokens, y, output=output)

    def extract(self, tokens, tags=None, output='plain'):
        '''
            tokens: list of token
            tags: list of tag
        '''
        if tags:
            X_features = self.transform_one(list(zip(tokens, tags)))
        else:
            X_features = self.transform_one(tokens)

        y = self.crf_model.predict_single(X_features)

        return self.pretty_entities(tokens, y, output=output)



if __name__ == '__main__':
    # train model
    import jieba.posseg as pseg
    from simcrf import SimCRF
    from simcrf.utils import read_iob_file
    
    ner = SimCRF()

    # sents = read_iob_file('../data/crf_4689_4_entity.txt')
    # X_train = [sent2features(s) for s in sents]
    # y_train = [s[2] for s in sents]
    X_train = [
        [
            ('打印机', 'n'), ('采购', 'v'), ('品目', 'n'), ('采购', 'v'), ('单位', 'n'), ('曲周县', 'nr'), ('职业', 'n'), ('技术', 'n'), ('教育', 'vn'), ('中心', 'n'), ('行政区域', 'n'), ('曲周县', 'nr'), ('公告', 'n'), ('时间', 'n')
        ],
        [
            ('打印机', 'n'), ('采购', 'v'), ('品目', 'n'), ('采购', 'v'), ('单位', 'n'), ('曲周县', 'nr'), ('职业', 'n'), ('技术', 'n'), ('教育', 'vn'), ('中心', 'n'), ('行政区域', 'n'), ('曲周县', 'nr'), ('公告', 'n'), ('时间', 'n')
        ]
    ]

    y_train = [
        ['O','O','O','O','O','B','I','I','I','I','O','O','O','O'],
        ['O','O','O','O','O','B','I','I','I','I','O','O','O','O']
    ]

    X_train = ner.transform(X_train)
    ner.fit(X_train, y_train)

    # save model
    ner.save('~/crf_test.pkl')

    # load model
    from pathlib import Path
    ner = SimCRF(crf_model_path=Path('.').parent.absolute() / 'data' / 'crf_bid_4_entities.pkl')

    # extract
    text = r'''
    　哈尔滨工业大学招标与采购管理中心受总务处的委托，就哈尔滨工业大学部分住宅小区供热入网项目（项目编号：GC2017DX035）组织采购，评标工作已经结束，中标结果如下：

    一、项目信息

    项目编号：GC2017DX035

    项目名称：哈尔滨工业大学部分住宅小区供热入网

    项目联系人：李占奎 王 吉

    联系方式：电话： 0451-86417953 13936645563

    

    二、采购单位信息

    采购单位名称：总务处

    采购单位地址：哈尔滨市南岗区西大直街92号

    采购单位联系方式：孔繁武 0451-86417975

    

    三、项目用途、简要技术要求及合同履行日期：

    见结果公示

    

    四、采购代理机构信息

    采购代理机构全称：哈尔滨工业大学招标与采购管理中心

    采购代理机构地址：哈尔滨市南岗区西大直街92号哈尔滨工业大学行政办公楼203房间

    采购代理机构联系方式：李占奎 王 吉 电话： 0451-86417953 13936645563
    '''
    sent = [tuple(pair) for pair in pseg.cut(text)]
    ret = ner.extract_taggedtokens(sent)

    print(ret)