# -*- coding: utf-8 -*-
# @Author: mithril

from __future__ import unicode_literals, print_function, absolute_import, division


from simcrf import SimCRF
from simcrf.features import sent2labels, CrfTransformer
from simcrf.utils import pos_cut
from lxml import html
from textminer.project.bid.crf import bid_crf_extractor
import unicodedata

import regex as re

def extract_project(content):
    return bid_crf_extractor.extract(content).get('project')


def html2text(htmlstr):

    h = html.fromstring(htmlstr)

    for br in h.xpath("//br"):
        br.tail = "\n" + br.tail if br.tail else "\n"

    for p in h.xpath("//p"):
        p.text = "\n" + p.text if p.text else "\n"

    for div in h.xpath("//div"):
        div.text = "\n" + div.text if div.text else "\n"

    return h.text_content()


def clean_text(text):
    content = unicodedata.normalize('NFKC', text)
    content = content.replace('&amp;nbsp', '')
    content = content.replace('&nbsp', '')
    content = content.replace('\r', '\n')
    content = re.sub(r'\n\s*', '\n', content)
    content = re.sub(r'  *', ' ', content)
    return content


def test():

    text = '''
<div><p>　　德汇工程管理(北京)有限公司受北京市东城区人民政府东花市街道办事处委托，根据《中华人民共和国政府采购法》等有关规定，现对2018年东花市街道“百街千巷”环境整治提升工程（全过程）进行竞争性磋商招标，欢迎合格的供应商前来投标。</p><p> </p><p><strong>项目名称：</strong>2018年东花市街道“百街千巷”环境整治提升工程（全过程）</p><p><strong>项目编号：</strong>TAHP-2018-ZB-034</p><p><strong>项目联系方式：</strong></p><p>项目联系人：梁辰</p><p>项目联系电话：15201579156</p><p> </p><p></p><p><strong>采购单位联系方式：</strong></p><p>采购单位：北京市东城区人民政府东花市街道办事处</p><p></p>  <p>采购单位地址：北京市东城区东花市北里西区3号楼</p><p>采购单位联系方式：张科长，13720079926</p><p> </p><p></p><p><strong>代理机构联系方式：</strong></p><p>代理机构：德汇工程管理(北京)有限公司</p><p>代理机构联系人：梁辰  15201579156</p><p>代理机构地址： 北京市丰台区汽车博物馆东路6号院G座盈坤世纪7层702</p><p></p>       <p> </p><p><strong>一、采购项目的名称、数量、简要规格描述或项目基本概况介绍：</strong></p> <p>项目名称：2018年东花市街道“百街千巷”环境整治提升工程（全过程）<br>项目概况：<br>1、采购内容：街道分指组织实施的（1）整治项目、提升项目或者整治提升“一体化”项目；（2）其他整治项目（“百街千巷”环境整治提升2018年任务街巷台账以外区域的环境整治项目）的全过程管理服务；<br>2、工程投资金额：647.31万元；<br>3、招标控制取费费率：依据财政部《关于印发基本建设项目建设成本管理规定的通知》（财建[2016]504号），结合全过程管理单位工作职责，采取阶梯取费费率。<br>4、资金来源：财政性资金。</p><p><strong>二、对供应商资格要求（供应商资格条件）:</strong></p><p>供应商参加本次政府采购活动应具备下列条件：1、符合《政府采购法》第二十二条规定的条件1)具有独立承担民事责任的能力2)具有良好的商业信誉和健全的财务会计制度3)具有履行合同所必须的设备和专业技术能力4)具有依法缴纳税收和社会保障资金的良好记录5)参加本次磋商活动前三年内，在经营活动中没有重大违法违规记录2、本项目不接受联合体磋商。3、供应商必须到采购代理机构购买磋商文件（即“采购文件”，下同）并登记备案。未经向采购代理机构购买磋商文件并登记备案的供应商将被拒绝。4、在中华人民共和国境内注册，有能力提供本项目服务的供应商，包括法人、其他组织、自然人。5、符合《财政部关于在政府采购活动中查询及使用信用记录有关问题的通知》（财库〔2016〕125号）的相关要求 。</p><p> </p><p><strong>三、磋商和响应文件时间及地点等:</strong></p><p>预算金额：0.0 万元（人民币）</p><p>谈判时间：2018年04月13日 09:00</p><p>获取磋商文件时间：2018年04月03日 09:00 至 2018年04月10日 16:00(双休日及法定节假日除外)</p><p>获取磋商文件地点：北京市丰台区汽车博物馆东路6号盈坤世纪G座702</p><p>获取磋商文件方式：（1）法人授权委托书（原件）、被授权人身份证复印件；（2）法人营业执照副本复印件（复印件须加盖单位公章）</p><p>磋商文件售价：200.0  元（人民币）</p><p>响应文件递交时间：2018年04月13日 08:50 至 2018年04月13日 09:00(双休日及法定节假日除外)</p><p>响应文件递交地点：北京市丰台区汽车博物馆东路6号院盈坤世纪G座7层702室第一会议室</p><p>响应文件开启时间：2018年04月13日 09:00</p><p>响应文件开启地点：北京市丰台区汽车博物馆东路6号院盈坤世纪G座7层702室第一会议室</p><p> </p><p><strong>四、其它补充事宜：</strong></p><p></p><p>无</p><p> </p><p><strong>五、项目联系方式：</strong></p><p>项目联系人：梁辰</p><p>项目联系电话：15201579156</p><p> </p><p><strong>六、采购项目需要落实的政府采购政策：</strong></p> <p></p><p>本项目需落实的中小微型企业扶持等相关政府采购政策详见磋商文件</p><p> </p><p></p>    <p> </p>           <p></p>
            </div>
    '''

    print(extract_project(text))
    
    t = html.fromstring(text).text_content()
    t = clean_text(t)
    print(t)
    
    print(extract_project(t))

    print(extract_project(html2text(text)) )

    # x = bid_crf_extractor.models['project']
    # tf = CrfTransformer(tokenizer=pos_cut)
    # xf = tf.transform_one(text)
    # xf = list(xf)
    # print(x.crf_model.predict_single(xf))

    # xf = tf.transform_one(html.fromstring(text).text_content())
    # xf = list(xf)
    # print(x.crf_model.predict_single(xf))

    # xf = tf.transform_one(html2text(text))
    # xf = list(xf)
    # print(x.crf_model.predict_single(xf))



test()