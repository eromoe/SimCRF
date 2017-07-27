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

from .features import sent2features, tokens2offsets, token2features, tag2features


class SimCRF(object):

    def __init__(self, crf_model=None, crf_model_path=None):
        self.crf_model = crf_model
        if not crf_model and crf_model_path:
            with open(crf_model_path, 'rb') as f:
                self.crf_model = pickle.load(f)

    def transform(self, sents):
        return [sent2features(s) for s in sents]

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

    def entity_extract(self, sent, tags=None, output='plain'):
        '''
            sent: list of (token, tag) , or list of token
            tags: list of tags
            offsets: each token's offset
        '''
        assert output in ('token', 'plain', 'offset')

        zipped = not isinstance(sent[0], basestring)
        
        if zipped:
            tokens = [token for token, tag in sent]
            tags = [tag for token, tag in sent]
        else:
            tokens = sent
            

        if output == 'offset':
            offsets = tokens2offsets(tokens)
                
        if zipped:
            X_features = sent2features(tokens)
        else:
            if tags:
                X_features = sent2features(list(zip(tokens, tags)))
            else:
                X_features = [token2features(tokens, i) for i in range(len(tokens))]

        y = self.crf_model.predict_single(X_features)

        entities = []
        start = None
        curtag = None

        for cursor, tag in enumerate(y):
            if tag.endswith('B'):
                start = cursor
                curtag = tag.split('-')[0]
            elif start and tag == 'O':
                # entities.append((start, cursor))
                if output == 'plain':
                    if zipped:
                        entities.append((curtag, ''.join(i[0] for i in tokens[start:cursor])))
                    else:
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


if __name__ == '__main__':
    # train model
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

    ner.fit(X_train, y_train)

    # save model
    ner.save('~/crf_test.pkl')

    # load model
    ner = SimCRF(crf_model_path='~/crf_test.pkl')

    # extract
    text = r'''福建康泰招标有限公司受福州市仓山区第六中心小学的委托，就VR教室设备采购项目项目（项目编号：KTZB-2017017）组织采购，评标工作已经结束，中标结果如下： 一、项目信息项目编号：KTZB-2017017项目名称：VR教室设备采购项目项目联系人：周女士联系方式：0591-87803505 二、采购单位信息采购单位名称：福州市仓山区第六中心小学采购单位地址：仓山区采购单位联系方式：邱主任18350117477 三、项目用途、简要技术要求及合同履行日期：项目用途：教学简要技术要求：中标供应商负责组织专业技术人员进行设备安装调试，采购人应提供必须的基本条件和专人配合，保证各项安装工作顺利进行。中标供应商应协调配合完成项目的安装、调试且直至验收合格，使用正常。其他详见招标文件第三章招标内容及要求。  四、采购代理机构信息采购代理机构全称：福建康泰招标有限公司采购代理机构地址：福州市鼓楼区湖东路169号中闽天骜大厦13层采购代理机构联系方式：周女士0591-87803505 五、中标信息招标公告日期：2017年03月14日中标日期：2017年04月06日总中标金额：24.88 万元（人民币）中标供应商名称、联系地址及中标金额：序号          中标供应商名称         中标供应商联系地址           中标金额(万元)        1           福州裕兴网络科技有限公司            福建省福州市仓山区城门镇南江滨西大道198号福州海峡国际会展中心地下一层东区办公室中心A-056（自贸实验区内）            24.88        评审专家名单：陈晓英（组长）、林雪山、陈景瑞、陈同熙、邱其林（业主评委） 中标标的名称、规格型号、数量、单价、服务要求：中标标的名称：VR云端集控系统等其他详见招标文件规格型号：动感课堂VACYC01等其他详见投标文件数量：1批单价：24.88万元服务要求： 中标供应商应对本次投标设备承诺自验收合格后免费上门保修一年，三个月包换,其它详见招标文件要求。 六、其它补充事宜                  采购人和评审专家的推荐意见（采用书面推荐供应商参加采购活动的需填）： 经评标委员会评议，各投标人的资格及符合性均符合招标文件要求。  '''
    tokens = [tuple(pair) for pair in pseg.cut(text)]
    ret = ner.entity_extract(tokens)

    print(ret)