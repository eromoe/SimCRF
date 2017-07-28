# -*- coding: utf-8 -*-
# @Author: mithril
# @Date:   2017-04-26 11:16:51
# @Last Modified by:   mithril
# @Last Modified time: 2017-04-26 13:40:12

from __future__ import unicode_literals, print_function, absolute_import

import time
import os
import codecs
import glob
import itertools
import regex as re
import unicodedata
# import jieba
# jieba.suggest_freq(('医院', '地址'), True)
import jieba.posseg as pseg

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import ujson as json
except ImportError:
    import json

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def jieba_tag(word):
    from jieba.posseg import dt
    try:
        return dt.word_tag_tab['word']
    except KeyError:
        return 'x'


def pos_cut(sentence, HMM=True):
    start = 0
    for w in pseg.cut(sentence, HMM=HMM):
        yield w, (start, start+len(w.word))
        start += len(w.word)



def mark_iob(content, entity_offsets, full_text=False, iob=True):
    '''
    return  list of list , [[]]
    '''

    begin = 0
    last = None
    hit = False
    results = list(pos_cut(content))

    # 必须这里全部标上
    matched_blocks = []

    for t, offset in results: t.new_flag = 'O'

    for tk_idx, (t, offset) in enumerate(results):
        pretag = entity_offsets[begin][1]

        if entity_offsets[begin-1][0][1] > entity_offsets[begin][0][0] >= entity_offsets[begin-1][0][0]:
            # 存在重叠
            begin += 1
            continue

        if offset[0] == entity_offsets[begin][0][0]:
            t.new_flag = pretag+'-B' if iob else pretag

            last = entity_offsets[begin][0][1]

            # 一个单词刚好完整匹配
            if offset[1] == last:
                matched_blocks.append(results[tk_idx-5 if tk_idx -5 >0 else 0: tk_idx + 5 if tk_idx + 5 < len(results) else len(results)])
                last = None
                begin +=1

            cur_start = tk_idx

        elif last:

            if offset[1] < last:
                t.new_flag = pretag+ '-I' if iob else pretag
            elif offset[1] == last:
                t.new_flag = pretag+ '-I' if iob else pretag
                    #t.new_flag == 'F'
                matched_blocks.append(results[cur_start-5 if cur_start -5 >0 else 0: tk_idx + 5 if tk_idx + 5 < len(results) else len(results)])

                last = None
                begin +=1
            elif offset[0] < last < offset[1]:

                token_l = t.word[offset[1] - last:]
                token_r = t.word[:offset[1] - last]
                tag_l = jieba_tag(token_l)
                tag_r = jieba_tag(token_r)

                offset_l = (offset[0], last)
                offset_r = (last, offset[1])

                from jieba.posseg import pair

                t_l = pair(token_l, tag_l)
                t_r = pair(token_r, tag_r)
                t_l.new_flag = pretag+ '-I' if iob else pretag
                t_r.new_flag = 'O'

                tagged_block = results[cur_start-5 if cur_start -5 >0 else 0: tk_idx] + \
                    [(t_l, offset_l), (t_r, offset_r)] + \
                    results[tk_idx+1:tk_idx + 5 if tk_idx + 5 < len(results) else len(results)]

                matched_blocks.append(tagged_block)
                # 结巴分词出错， 比如 xxx 医院 地址  ， 被切成 xxx 医院地址 ， 医院地址 长度超过了 医院
                # raise Exception(t, offset, entity_offsets[begin])

        if begin >= len(entity_offsets):
            break

        # if 'new_flag' not in t.__dict__:
        #     t.new_flag = 'O'

    if full_text:
        return [results]
    else:
        return matched_blocks



def bid_json_2_crf_trainset(corpus_dir, output_dir, output_name=None, output_type='json', train_tags=['contact','proxy_name','project_name','buyer'], full_text=False):
    '''
    # 注意 中国招标网   在项目联系人， 代理机构名称等上 都有可能是 “详见公告正文”, “详见公告”
    # 代理机构名称 见到过 电话号码， 或者有标点符号包含在内的
    # 有的是一整段话 “采购代理机构：上海容基工程项目管理有限公司青海分公司 联系人：董女士 联系电话：0971-8166798-8088”
    '''

    ensure_dir(output_dir)

    file_paths = []
    file_paths.extend(glob.glob(os.path.join(corpus_dir, '*.json')))
    lst = []
    num = 1

    matched_blocks = []

    for p in file_paths:
        print(p)
        with codecs.open(p, 'r', 'utf-8') as f:
            for idx, s in enumerate(f):
                item = json.loads(s)

                title =item['result']['title']
                content =item['result']['text']
                url =item['result']['url']
                contact =item['result']['extra'].get('项目联系人', '').strip('：:\t')
                proxy_name =item['result']['extra'].get('代理机构名称', '').strip('：:\t')
                project_name =item['result']['extra'].get('采购项目名称', '').strip('：:\t')
                buyer =item['result']['extra'].get('采购单位', '').strip('：:\t')

                # if proxy_name in ('详见公告正文', '详见公告'):
                #     proxy_name = ''

                # if no_blank:
                # content = re.sub(r"[^\S\n]+", "", content)
                content = unicodedata.normalize('NFKC', content)
                content = re.sub(r'\n\s*', '', content)
                content = re.sub(r' ', '', content)

                # 替换掉 \n 可能会导致训练精度下降
                # content = re.sub(r"\s", " ", content)

                if '详见' in proxy_name:
                    proxy_name = ''
                elif len(contact) < 6 :
                    # 错误内容 都很短，比如  见公告, /
                    project_name = ''

                if '详见' in buyer:
                    buyer = ''

                if '详见' in project_name:
                    project_name = ''
                elif len(project_name) < 7 :
                    # 错误内容 都很短，比如 /
                    project_name = ''

                if '详见' in contact:
                    contact = ''
                elif len(contact) < 2 :
                    contact = ''
                    # 空格 + 电话， 空格 + 另一个名字， 逗号 + 名字
                    # elif ' ' in contact:
                    #     contact = contact.split()[0]
                    #     if len(contact) > 4:
                    #         contact = ''
                entities = []
                for tag in train_tags:
                    if tag == 'contact':
                        entities.append((contact, 'CT-'))
                    elif tag == 'proxy_name':
                        entities.append((proxy_name, 'PX-'))
                    elif tag == 'project_name':
                        entities.append((project_name, 'PJ-'))
                    elif tag == 'buyer':
                        entities.append((buyer, 'BY-'),)

                entity_offsets = []
                for entity, pretag in entities:
                    if entity:
                        for i in re.finditer(re.escape(entity), content):
                            entity_offsets.append((i.span(), pretag))

                if not entity_offsets:
                    continue

                entity_offsets = sorted(entity_offsets, key=lambda x: x[0][0])

                matched_blocks.extend(mark_iob(content, entity_offsets, full_text))

                # for sent in results2:
                #     output_path = os.path.join(output_dir, '%09d.txt' % num)
                #     num += 1
                #     with codecs.open(output_path, 'w', 'utf-8') as f:
                #         f.write('\n'.join([' '.join((t.word, t.flag, t.new_flag if 'new_flag' in t.__dict__ else 'O')) for t, _ in sent]))

                if idx % 500 == 0:
                    print('processed:%d' % idx)

        output_path = os.path.join(output_dir, output_name if output_name else 'crf_all_%s_%s.%s' % ( 'full' if full_text else 'parts' , idx, output_type))

        # 使用txt 分割的话，会导致需要把 \n 替换掉，否则会多出分行，无法训练带 \n的数据，所以转json
        with codecs.open(output_path, 'w', 'utf-8') as f:
            if output_type == 'json':
                data = [[(t.word, t.flag, t.new_flag) for t, _ in sent] for sent in matched_blocks]
                f.write(json.dumps(data))
            else:
                f.write('\n\n'.join(['\n'.join([' '.join((t.word, t.flag, t.new_flag)) for t, _ in sent]) for sent in matched_blocks]))

if __name__ == '__main__':

    corpus_dir = 'bid_json'
    output_dir = '.'
    bid_json_2_crf_trainset(corpus_dir, output_dir, full_text=False)