from nltk.corpus import wordnet as wn
import spacy

nlp = spacy.load('es_core_news_md') 
map_Spacy_to_wordnet_POS ={'PROPN': 'n', 'DET': 'r', 'ADJ': 'a', 'ADV': 'r', 'NOUN': 'n', 'DET': 'r', 'PROPN': 'n', 'VERB': 'v'}

def extract_sentences(text_list, source, vocab):
    sentences = []
    sent_index = 0
    docs = nlp.pipe(text_list)
    doc_index = 0
    for doc in docs:
        
        for  sent in doc.sents:
            tok_list = []
            tok_ignored = []
            for token in sent:
                if token.pos_ in map_Spacy_to_wordnet_POS.keys():
                    if token.text.lower() not in nlp.Defaults.stop_words and token.lemma_.lower() not in nlp.Defaults.stop_words:
                        if token.lemma_ not in vocab['tokens'].keys():
                            synsets = wn.synsets(token.lemma_,pos=map_Spacy_to_wordnet_POS[token.pos_],lang='spa')
                            tok = {'text':token.text,'pos':token.pos_,'vector':token.vector if token.has_vector else [], 'synsets':synsets,'hit_'+source:[sent_index]}
                            vocab['tokens'][token.lemma_] = tok 
                        else:
                            if token.lemma_ not in tok_list:
                                if 'hit_'+source in vocab['tokens'][token.lemma_].keys():
                                    vocab['tokens'][token.lemma_]['hit_'+source].append(sent_index)
                                else:
                                    vocab['tokens'][token.lemma_]['hit_'+source] = [sent_index]

                        tok_list.append(token.lemma_)
                    else:
                        tok_ignored.append(token.lemma_)
                        if token.lemma_ not in vocab['ignored_tokens'].keys():
                            synsets = wn.synsets(token.lemma_,pos=map_Spacy_to_wordnet_POS[token.pos_],lang='spa')
                            tok = {'text':token.text,'pos':token.pos_,'vector':token.vector if token.has_vector else [], 'synsets':synsets,'hit_'+source:[sent_index]}
                            vocab['ignored_tokens'][token.lemma_] = tok 
                        else:
                            if token.lemma_ not in tok_ignored:
                                vocab['ignored_tokens'][token.lemma_]['hit_'+source].append(sent_index)
            if len(tok_list) > 0:
                idea = dict()
                idea['opinion'] = doc_index
                idea['content'] = sent.text
                idea['tokens'] = tok_list
                idea['ignored_tokens'] = tok_ignored
                sentences.append(idea)
                sent_index += 1
        doc_index+=1    
    return sentences


def extract_nouns_from_news(news, vocab, source='news'):
    nouns = []
    doc = nlp(news)
    for token in doc:
        if token.pos_ == ('NOUN' or 'PROPN'):
            if token.lemma_ not in vocab['tokens'].keys():
                synsets = wn.synsets(token.lemma_,pos=map_Spacy_to_wordnet_POS[token.pos_],lang='spa')
                tok = {'text':token.text,'pos':token.pos_,'vector':token.vector if token.has_vector else [], 'synsets':synsets,'hit_'+source:[0]}
                vocab['tokens'][token.lemma_] = tok 
            else:
                if token.lemma_ not in nouns:
                    vocab['tokens'][token.lemma_]['hit_'+source] = [0]

            nouns.append(token.lemma_)
    
    return [{'content': news, 'tokens':nouns}]