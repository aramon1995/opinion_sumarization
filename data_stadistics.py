from load_data import load_csv
import nltk
import os

def opinion_statistics(op_dir):
    cant_token = 0
    cant_sents = 0
    opinions = load_csv(op_dir)
    texts = [o['content']for o in opinions]
    opinions = [[[token for token in nltk.tokenize.word_tokenize(s,language='spanish')]for s in nltk.sent_tokenize(text,language='spanish')] for text in texts]
    for opinion in opinions:
        cant_sents += len(opinion)
        for sent in opinion:
            cant_token += len(sent)
    return(opinions,cant_sents,cant_token)

def corpus_statistics(dir):
    files = os.listdir(dir)
    files.sort()
    opinions_data = []
    for file in files:
        opinions_data.append(opinion_statistics(os.path.join(dir,file)))
    cant_news = len(opinions_data)
    cant_op = 0
    cant_sentences = 0
    cant_terms = 0
    for od in opinions_data:
        cant_op += len(od[0])
        cant_sentences += od[1]
        cant_terms += od[2]
    
    cant_prom_op_for_news = cant_op/cant_news
    prom_sent_for_op = cant_sentences/cant_op
    prom_term_for_op = cant_terms/cant_op
    print('CANTIDAD DE NOTICIAS: ',cant_news)
    print('CANTIDAD DE OPINIONES: ',cant_op)
    print('CANTIDAD DE ORACIONES: ',cant_sentences)
    print('CANTIDAD DE TERMINOS: ',cant_terms)
    print('PROMEDIO DE OPINIONES POR NOTICIA: ',cant_prom_op_for_news)
    print('PROMEDIO DE ORACIONES POR OPINION: ',prom_sent_for_op)
    print('PROMEDIO DE TERMINOS POR ORACION: ',prom_term_for_op)



