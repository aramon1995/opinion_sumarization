import score_sentences
import score_topic

def filter_sentences_for_polarity(sentences, threshold=1):
    return list(filter(lambda s:s['polarity'][0] != 0 or s['polarity'][1] != 0, sentences))



def filter_senteces_for_news_similarity(vocab, sentences,source,news,news_source='news',threshold=0.5):
    score_sentences.sentence_refers_to_news(sentences,source,news,vocab)
    return list(filter(lambda s:s['score']>threshold,sentences))




def filter_topics_for_news_similarity(threshold, vocab, topics, source, news, unit = 'sentences'):
    if unit == 'sentences':
        score_topic.sentences_refers_to_news(news,topics,source,vocab)
        return list(filter(lambda t:t['score']>threshold,topics))
    
    if unit == 'words':
        score_topic.words_refers_to_news(news,topics,source,vocab)
        return list(filter(lambda t:t['score']>threshold,topics))
        