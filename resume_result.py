import os
from os.path import join
from os import listdir


files = listdir('.')
print(files)
out = open('RESUMEN','w')
for f in files:
    if f == '1README.txt' or f == 'data.py':
        continue
    text = open(f,'r')
    lines = text.readlines()
    js_topic = [ float(line.rstrip().split(': ')[1]) for line in lines if 'jensen shannon divergence respect to topic' in line]
    js_news = [float(line.rstrip().split(': ')[1]) for line in lines if 'jensen shannon divergence respect to news' in line]
    js_opinions = [float(line.rstrip().split(': ')[1]) for line in lines if 'jensen shannon divergence respect to all opinions' in line]
    js_topic_sum = 0
    js_news_sum = 0
    js_opinions_sum = 0
    for i in range(len(js_topic)):
        js_topic_sum += js_topic[i]
        js_opinions_sum += js_opinions[i]
        js_news_sum += js_news[i]
    out.write(f.split('.')[0]+'\n')
    out.write(' js jensen shannon divergence respect to topic: '+str(js_topic_sum/len(js_topic))+'\n')
    out.write(' js jensen shannon divergence respect to all opinions: '+str(js_opinions_sum/len(js_opinions))+'\n')
    out.write(' js jensen shannon divergence respect to news: '+str(js_news_sum/len(js_news))+'\n')
    out.write('--------------------------------------------------------------------------------\n\n')
out.close()