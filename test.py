# -*- coding: utf-8 -*-
# @Author: mithril

from __future__ import unicode_literals, print_function, absolute_import, division



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
    ner.save('/tmp/crf_test.pkl')

    # load model
    from pathlib import Path
    ner = SimCRF(crf_model_path=str(Path('.').parent.absolute() / 'data' / 'crf_bid_4_entities.pkl'))

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