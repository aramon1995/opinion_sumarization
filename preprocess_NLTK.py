import treetaggerwrapper as ttwp
import nltk
import os
from os.path import join
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from gensim.models import KeyedVectors


vectors = KeyedVectors.load_word2vec_format('resources/word_vectors/embeddings-m-model.vec')
map_TreeTagger_to_wordnet_POS ={'ADJ': 'a', 'ADV': 'r', 'NC': 'n', 'NMEA': 'n', 'NMON': 'n', 'NP': 'n', 'V': 'v'}

def extract_sentences(text_list, source, vocab):
    tagger = ttwp.TreeTagger(TAGLANG='es',TAGDIR=join(os.getcwd(),'resources/tree_tagger/'))
    sentences = []
    sent_index = 0
    doc_index = 0
    for doc in text_list:
        sents = nltk.tokenize.sent_tokenize(doc,language = 'spanish')
        for  sent in sents:
            tokens = tagger.tag_text(sent.lower())
            tokens = ttwp.make_tags(tokens)
            tok_list = []
            tok_ignored = []
            for token in tokens:
                if type(token).__name__ == 'NotTag':
                    continue
                if token[1] in map_TreeTagger_to_wordnet_POS.keys() or token[1][0] in map_TreeTagger_to_wordnet_POS:
                    if token[0] not in stopwords.words('spanish') and token[2] not in stopwords.words('spanish'):
                        if token[2] not in vocab['tokens'].keys():
                            pos = 'v' if token[1][0] == 'V' else map_TreeTagger_to_wordnet_POS[token[1]]
                            synsets = wn.synsets(token[2],pos=pos,lang='spa')
                            vector = []
                            try:
                                vector = vectors[token[2]]
                            except:
                                pass
                            tok = {'text':token[0],'pos':token[1],'vector':vector, 'synsets':synsets,'hit_'+source:[sent_index]}
                            vocab['tokens'][token[2]] = tok 
                        else:
                            if token[2] not in tok_list:
                                if 'hit_'+source in vocab['tokens'][token[2]].keys():
                                    vocab['tokens'][token[2]]['hit_'+source].append(sent_index)
                                else:
                                    vocab['tokens'][token[2]]['hit_'+source] = [sent_index]

                        tok_list.append(token[2])
                    else:
                        tok_ignored.append(token[2])
                        if token[2] not in vocab['ignored_tokens'].keys():
                            pos = 'v' if token[1][0] == 'V' else map_TreeTagger_to_wordnet_POS[token[1]]
                            synsets = wn.synsets(token[2],pos=pos,lang='spa')
                            vector = []
                            try:
                                vector = vectors[token[2]]
                            except:
                                pass
                            tok = {'text':token[0],'pos':token[1],'vector':vector, 'synsets':synsets,'hit_'+source:[sent_index]}
                            vocab['ignored_tokens'][token[2]] = tok 
                        else:
                            if token[2] not in tok_ignored:
                                vocab['ignored_tokens'][token[2]]['hit_'+source].append(sent_index)
            if len(tok_list) > 0:
                idea = dict()
                idea['opinion'] = doc_index
                idea['content'] = sent
                idea['tokens'] = tok_list
                idea['ignored_tokens'] = tok_ignored
                sentences.append(idea)
                sent_index += 1
        doc_index+=1    
    return sentences


def extract_nouns_from_news(news, vocab, source='news'):
    nouns = []
    tagger = ttwp.TreeTagger(TAGLANG='es',TAGDIR=join(os.getcwd(),'resources/tree_tagger/'))
    doc = tagger.tag_text(news.lower())
    doc = ttwp.make_tags(doc)
    for token in doc:
        if type(token).__name__ == 'NotTag' or token[1] not in map_TreeTagger_to_wordnet_POS.keys():
            continue
        pos = map_TreeTagger_to_wordnet_POS[token[1]]
        if pos == 'n':
            if token[2] not in vocab['tokens'].keys():
                synsets = wn.synsets(token[2],pos=pos,lang='spa')
                vector = []
                try:
                    vector = vectors[token[2]]
                except:
                    pass
                tok = {'text':token[0],'pos':pos,'vector':vector, 'synsets':synsets,'hit_'+source:[0]}
                vocab['tokens'][token[2]] = tok 
            else:
                if token[2] not in nouns:
                    vocab['tokens'][token[2]]['hit_'+source] = [0]

            nouns.append(token[2])
    
    return [{'content': news, 'tokens':nouns}]