import csv
import os
from os.path import join
from os import listdir
import codecs


def proc(dir, delimiter):
    
    data = []
    with open(dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        # csv_reader.__next__()
        for row in csv_reader:
            try:
                r = {'content':row[3].lower(),'author':row[1],'date':row[2]}
            except:
                print(dir)
                print(row)
            if r not in data:
                data.append(r)
    return data


def clean_news(dir):
    news = ''
    with open(dir, 'r') as file_news:
        lines = file_news.readlines()
        for line in lines:
            news += ' '+line.rstrip()
    return news





path = 'data/etecsa/opinions/dirty/'
files = listdir(path)
print(files)
for f in files:
    c = proc(join(path,f),',')
    out = open(file=join('data/etecsa/opinions/',f),mode='w')
    count = 1
    for row in c:
        out.write(str(count)+','+row['author']+','+row['date']+',"'+row['content']+'"\n')
        count+=1
    out.close()

# path = 'data/clean/news/'
# files = listdir(path)
# print(files)
# for f in files:
#     n = clean_news(join(path,f))
#     out = open(file=join('data/news/',f),mode='w')
#     out.write(n)
#     out.close()