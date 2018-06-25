import pandas as pd
import numpy as np

df = pd.read_csv('../data/chat.csv')[['session_id','user_id','send','zhaunchu','repeat','sku','content']]
UNK='UNK'
temp = 0
allpairs = {}
all_session = []
save_file_path = '../data/new_group_np'

def df2pair(temp):
    now_chat = 0
    now_str = ''
    pair = []
    for i in range(len(temp)):
        if(int(temp.iloc[i].send) == int(now_chat)):
            now_str = now_str + temp.iloc[i].content
        else:
            pair.append(now_str)
            now_chat = temp.iloc[i].send
            now_str = temp.iloc[i].content
        if(i == len(temp)-1):
            pair.append(now_str)
    return pair

import re
import jieba

def translate(str):
    line = str.strip().encode().decode('utf-8', 'ignore')
    p2 = re.compile(u'[^\u4e00-\u9fa5]')
    zh = "".join(p2.split(line)).strip()
    zh = "".join(zh.split())
    outStr = zh  # 经过相关处理后得到中文的文本
    return outStr

def ch_normalizeString(item):
    rtn = []
    for s in item:
        if(type(s)==str):
            s = s.encode().decode("utf8")
            s = re.sub("[A-Za-z0-9\s+\.\!\/_,$%^*(+\"\']+|[+——《》【】：”“！-，。？?、~@#￥%……&*（）]+:".encode().decode("utf8"),
                       "".encode().decode("utf8"), s)
            s = translate(s)
            s = ' '.join(list(jieba.cut(s, cut_all=False)))
            rtn.append(s)
    return rtn

def replace_lowf(min_frequency,texts):
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token if frequency[token] > min_frequency else UNK for token in text] for text in texts]
    #print(len([i for i in frequency if frequency[i] >= min_frequency]))

for idx,item in df.groupby(df.session_id):
    if(len(idx)>1):
        allpairs[idx] = item

for x in allpairs.values():
    try:
        all_session.append(df2pair(x))
    except:
        pass

for i,item in enumerate(all_session):
    all_session[i] = ch_normalizeString(item)
replace_lowf(2,all_session)
np_all_session = np.array(all_session)
np.save(save_file_path,np_all_session)
