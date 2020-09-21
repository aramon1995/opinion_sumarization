def proc(dir, delimiter):
    import csv
    data = []
    with open(dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            r = {'content':row[3].lower(),'author':row[2],'date':row[4]}
            if r not in data:
                data.append(r)
    return data


def extract_opinions(dir,delimiter=','):
    import csv
    import nltk
    import string
    data = []
    rows = []
    with open(dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        csv_reader.__next__()
        for row in csv_reader:
            r = row[3].lower()
            if r in rows:
                continue
            rows.append(r)
            sent = nltk.tokenize.sent_tokenize(r,language='spanish')
            for s in sent:
                tok = nltk.word_tokenize(s,language='spanish')
                tok = list(filter(lambda x:x not in string.punctuation, tok))
                data.append(tok)
    return data

c = proc('data/web_scrapping/movil_tarifa.csv',',')
out = open(file='data/movil_tarifa_clean.csv',mode='w')
count = 1
for row in c:
    out.write(str(count)+','+row['author']+','+row['date']+',"'+row['content']+'"\n')
    count+=1
out.close()

def get_url(dir,delimiter = ','):
    import csv
    with open(dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        csv_reader.__next__()
        return csv_reader.__next__()[1]    


import os 

files = os.listdir('data/web_scrapping')
urls = []
urls_file = open('urls.txt','w')
for f in files:
    urls_file.write(get_url(os.getcwd()+'/data/web_scrapping/'+f)+'\n')
urls_file.close()