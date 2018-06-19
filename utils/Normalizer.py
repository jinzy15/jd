import re
import jieba


# def unicodeToAscii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn'
#     )

# Lowercase, trim, and remove non-letter characters
# def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub("([.!?])", "", s)
#     s = re.sub("[^a-zA-Z.!?]+"," ", s)
#     return s
def ch_normalizeString(s):
    s = s.encode().decode("utf8")
    s = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——《》【】：”“！-，。？?、~@#￥%……&*（）]+".encode().decode("utf8"),
               "".encode().decode("utf8"), s)
    s = ' '.join(list(jieba.cut(s, cut_all=False)))
    return s