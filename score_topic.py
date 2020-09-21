import similarities
import numpy as np

def words_refers_to_news(news, topics, source, vocab):
    similarity_topics_news = similarities.sentences_similarity(topics, 'topics_of_words_from_'+source, news, 'news', vocab)
    for i in range(len(topics)):
        topics[i]['score'] = similarity_topics_news[1][i][0]


def sentences_refers_to_news(news, topics, source, vocab):
    for t in topics:
        t['score'] = similarities.sentences_similarity(t['sentences'], 'topics_of_sentences_from_'+source, news,'news',vocab)[1].mean()


def words_most_popular(topics, source, vocab):
    scores = similarities.sentences_similarity(topics,'topics_of_words_from_'+source,topics,'topics_of_words_from_'+source,vocab)[1].mean(axis=1)
    for t in range(len(topics)):
        topics[t]['score'] = scores[t]


def sentences_most_popular(topics, source, vocab):
    similarity_matrix = np.ones((len(topics),len(topics)))
    for t in range(len(topics) - 1):
        for t1 in range(t+1, len(topics)):
            similarity_matrix[t,t1] = similarity_matrix[t1,t] = similarities.sentences_similarity(topics[t]['sentences'],'topics_of_sentences_from_'+source,topics[t1]['sentences'],'topics_of_sentences_from_'+source,vocab)[1].mean()
    
    for t in range(len(topics)):
        topics[t]['score'] = similarity_matrix[t].mean()
    


