# SimCRF

This project is aim to provide a super easy way to train crf model and extract entities from text.

[中文文档](https://github.com/eromoe/SimCRF/blob/master/README.CN.md)

## Installation

    pip install simcrf

## Training Data Format

crf usually use iob tagging (https://en.wikipedia.org/wiki/Inside_Outside_Beginning)

input data can be:

1. word, pos_tag, iob_tag
2. word,  iob_tag

iob_tag:

- I: inside of entity
- O: outside of entity
- B: beginning of entity

Example:

    打印机 n O
    采购 v O
    品目 n O
    采购 v O
    单位 n O
    曲周县 nr B
    职业 n I
    技术 n I
    教育 vn I
    中心 n I
    行政区域 n O
    曲周县 nr O
    公告 n O
    时间 n O

    技术 n I
    教育 vn I
    中心 n I
    采购 v O
    单位地址 n O
    曲周县 nr B
    职业 n I
    技术 n I
    教育 vn I
    中心 n I
    采购 v O
    单位 n O
    联系方式 l O
    18932708288 m O

    中心 n I
    采购 v O
    人 n O
    地址 n O
    ： x O
    曲周县 nr B
    职业 n I
    技术 n I
    教育 vn I
    中心 n I
    采购 v O
    人 n O
    联系方式 l O
    ： x O


## Usage

#### Train model:

    from simcrf import SimCRF

    ner = SimCRF()

    # note: also support only tokens
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

#### Save model

    ner.save('~/crf_test.pkl')

#### Load model

    ner = SimCRF(crf_model_path='~/crf_test.pkl')

#### Extract entities

To support different tokenizer, you need tokenize your text first and feed to crf model.

    import jieba.posseg as pseg
    ner = SimCRF(crf_model_path='xxxx.pkl')

    text = '''    　哈尔滨工业大学招标与采购管理中心受总务处的委托，就哈尔滨工业大学部分住宅小区供热入网项目（项目编号：GC2017DX035）组织采购，评标工作已经结束，中标结果如下：

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

    采购代理机构联系方式：李占奎 王 吉 电话： 0451-86417953 13936645563'''

    sent = [tuple(pair) for pair in pseg.cut(text)]
    ret = ner.extract(sent)

    print(ret)

#### Custom crfsuite model

SimCrf aim to provide a simple and easy way to train and extract entities.
It take off the feature trasfroming and trainning apart from you.So to customize crf model, you need train a sklearn-crfsuite model. You would change trainning parameter and generation of features yourself, and pass the model to SimCRF:

    from simcrf import SimCRF
    import sklearn_crfsuite

    crf_model = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf_model.fit(X_train, y_train)

    ner = SimCRF(crf_model)

    ret = ner.extract(sent)

sklearn-crfsuite docs: https://sklearn-crfsuite.readthedocs.io/

crfsuite docs: http://www.chokkan.org/software/crfsuite/manual.html

