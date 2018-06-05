# -*- coding: utf-8 -*-
# @Author: mithril

from __future__ import unicode_literals, print_function, absolute_import, division


from simcrf import SimCRF
from simcrf.utils import pos_cut


def test_transform_text(sents):
    ner = SimCRF(tokenizer=pos_cut)

    new_sents = [pos_cut(sent) for sent in sents]
    
    r1 = ner.transformer.transform(sents, tokenize=True)
    r1 = list(r1)

    o1 = ner.transformer.transform_one(sents[0], tokenize=True)
    o1 = list(o1)

    r2 = ner.transformer.transform(new_sents)
    r2 = list(r2)

    o2 = ner.transformer.transform_one(new_sents[0])
    o2 = list(o2)

    r3 = ner.transform(sents, tokenize=True)
    r3 = list(r3)

    r4 = ner.transform(new_sents)
    r4 = list(r4)

    assert r1 == r2
    assert r1 == r3
    assert r1 == r4
    assert o1 == o2

def test_train_predict(sents):
    ner = SimCRF()
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

    ner.extract_taggedtokens(pos_cut('测试测试测试测试测试测试测试'))

    try:
        ner.extract('测试测试测试测试测试测试测试')
    except AssertionError :
        pass
    except Exception as e:
        raise(e)
    else:
        raise AssertionError('Must failed here !')