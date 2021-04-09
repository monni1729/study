import os
import json


with open('word_label.txt', 'r') as wl:
    word_label_map = dict()
    while True:
        line = wl.readline()
        tt = line.split('\t')
        
        temp = tt[1]
        temp = temp.split('\n')[0]
        word_label_map[tt[0]] = temp
        if len(word_label_map) == 82114:
            break
       
with open('word_label_map.json', 'w') as jf:
    json.dump(word_label_map, jf)