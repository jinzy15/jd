
# coding: utf-8

import pandas as pd
import numpy as np

descprition = 'chat'
datafile = descprition+'.txt'

f = open('../data/'+datafile, 'r')
all = []
for line in f.readlines():
    tmp = line.split('\t')
    if(len(tmp)>6):
        tmp[6] = ''.join(tmp[6:])
        tmp = tmp[:7]
    all.append(tmp)
df = pd.DataFrame(all[1:],columns=['session_id','user_id','send','zhaunchu','repeat','sku','content'])
df.to_csv('../data/'+descprition+'.csv')


def df2pair(temp):
    now_chat = 0
    now_str = ''
    pair = [[],[]]
    for i in range(len(temp)):
        if(int(temp.iloc[i].send) == int(now_chat)):
            now_str = now_str + temp.iloc[i].content
        else:
            pair[int(now_chat)].append(now_str)
            now_chat = temp.iloc[i].send
            now_str = temp.iloc[i].content
        if(i == len(temp)-1):
            pair[int(now_chat)].append(now_str)
    return pair


# In[5]:
print('start to make pairs')

temp = 0
allpairs = {}
for idx,item in df.groupby(df.session_id):
    if(len(idx)>1):
        try:
            allpairs[idx] = df2pair(item)
        except:
            pass


# In[ ]:
import json
json.dump(allpairs, open('../data/temp/'+descprition+'_dict'+'.json', 'w'),ensure_ascii=False)

#
# import re
# def del_charfstr(s,c):
#     s = re.sub(c, "", s)
#     return s
#
#
# # In[ ]:
# print('start to make txt')
#
# pair_txt = ''
# for x in allpairs.values():
#     for i in range(min(len(x[0]),len(x[1]))):
#         try:
#             pair_txt = pair_txt+del_charfstr(x[0][i],'[\n\t]')+'\t'+ del_charfstr(x[1][i],'[\n\t]') + '\n'
#         except:
#             pass
#
# f = open('../data/'+descprition+'_pair'+'.txt', 'w') # 若是'wb'就表示写二进制文件
# f.write(pair_txt)
# f.close()
#
