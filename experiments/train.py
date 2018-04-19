# -*- coding: utf-8 -*-
# @Author: mithril
# @Date:   2017-04-11 11:12:03
# @Last Modified by:   mithril
# @Last Modified time: 2017-05-02 16:00:12

from __future__ import unicode_literals, print_function, absolute_import

import codecs
import unicodedata
import json
import pickle

import jieba
import jieba.posseg as pseg
import scipy.stats
from sklearn.cross_validation import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics


def train_from_file(input_path, output_path):

    if input_path.endswith('.txt'):
        with codecs.open(input_path, 'r', 'utf-8') as f:
            sents = f.read().split('\n\n')
            sents = list(map(lambda x: [i.split() for i in x]  , ( s.split('\n') for s in sents)))
    elif input_path.endswith('.json'):
        with codecs.open(input_path, 'r', 'utf-8') as f:
            sents = json.loads(f.read())
    else:
        raise NotImplementedError()

    X_train = [sent2features(s) for s in sents]
    y_train = [sent2labels(s) for s in sents]

    crf_model = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf_model.fit(X_train, y_train)

    with open(output_path, 'wb') as f:
        pickle.dump(crf_model, f)



if __name__ == '__mian__':

    crf_train_path = 'corpus/crf_all_12944.txt'
    with codecs.open(crf_train_path, 'r', 'utf-8') as f:
        sents = f.read().split('\n\n')

    all_sents = list(map(lambda x: [i.split() for i in x]  , ( s.split('\n') for s in sents)))

    train_sents , test_sents = train_test_split(all_sents, test_size=0.05)


    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]


    crf = sklearn_crfsuite.CRF(
        # algorithm='lbfgs',
        algorithm='l2sgd',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)


    labels = list(crf.classes_)
    labels.remove('O')
    print(labels)

    y_pred = crf.predict(X_test)
    metrics.flat_f1_score(y_test, y_pred,
                          average='weighted', labels=labels)

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))

    crf_model_path = os.path.join(os.path.dirname(__file__), 'data', 'crf_uni.pkl')

    with open(crf_model_path, 'wb') as f:
        pickle.dump(crf, f)
