import process_doc
import os
from gensim.models import Word2Vec
files = os.listdir('data/web_scrapping')
all_data = []
for f in files:
    data = process_doc.extract_opinions('data/web_scrapping/'+f)
    all_data += data

model = Word2Vec(all_data)
model.save('model.bin')

model.wv.save_word2vec_format('model.txt', binary=False)