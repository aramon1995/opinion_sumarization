import load_data
import preprocess
import similarities
from nltk.corpus import wordnet as wn
import extract_topic
import score_topic
import score_sentences
import polarity
import filters
import evaluation

data_test = ['la oracion es probar una oracion guerra. casa, la de es feo. probando como entiende que es el remos bueno','como si la escribiera alguien en un matar comentario, de esta forma se prueba la calidad de los resultados en las oraciones. como si la escribiera alguien en un comentario, de esta forma se prueba la calidad de los resultados en las oraciones.',
            'esta viene siendo probar oraciones oraciones oraciones mesa cama puerta forma la segunda horrible'
            ]

data_test1 = ['caminar andar','es ser no por casa','perro gato felino','probando como entiende que es el remos']


def test_preprocess():
    load_data.add_model_stopw(load_data.nlp)
    vocab = {'tokens':{},'ignored_tokens':{}}
    sentences = preprocess.extract_sentences(data_test,'list',vocab)
    for sentence in sentences:
        print(sentence['content'],'   ',sentence['tokens'],'   ',sentence['ignored_tokens'],'\n')

    print('\n\n')

    for v in vocab['tokens']:
        print(v,'  ',vocab['tokens'][v]['text'],'   ',vocab['tokens'][v]['pos'],'   ',vocab['tokens'][v]['synsets'],vocab['tokens'][v]['hit_list'],'\n')


def test_sim_words():
    load_data.add_model_stopw(load_data.nlp)
    vocab = {'tokens':{},'ignored_tokens':{}}
    sentences = preprocess.extract_sentences(data_test,'list',vocab)
    similarities.create_similarity_matrix_words_wn(vocab)
    topics = extract_topic.extract_topic_of_words(vocab,'opinions')
    for t in topics:
        print(t)
    
    print('\n\n')

    for t in vocab['words_in_similarity_matrix']:
        print(t,'   ',vocab['tokens'][t]['hit_topics_of_words_from_opinions'])

    
    for s in sentences:
        print(s['content'])

    sentence_topic = extract_topic.map_sentences_into_topics(sentences,'list',topics,'opinions',vocab)
    for s in sentence_topic:
        for t in s['sentences']:
            print(t['content'])
        print('\n\n')

        
def test_sim_sentences():
    import time
    start = time.time()
    print(start,'   start')
    load_data.add_model_stopw(load_data.nlp)
    opinons = load_data.load_csv('data/etecsa/opinions/movil_tarifa_clean.csv')
    data_test = [o['content'] for o in opinons]
    vocab = {'tokens':{},'ignored_tokens':{}}
    print(time.time() - start,'  loaded opinions')
    sentences = preprocess.extract_sentences(data_test,'list',vocab)
    print(time.time() - start,'  extracted sentences')
    # sentences1 = preprocess.extract_sentences(data_test1,'list1',vocab)
    similarities.create_similarity_matrix_words_wn(vocab)
    print(time.time() - start,'  similarity between words')
    # sent_similarity = similarities.sentences_similarity(sentences, 'list', sentences1, 'list1', vocab)

    # for sentence in sentences:
    #     print(sentence['content'],'   ',sentence['tokens'],'   ',sentence['ignored_tokens'],'\n')
    # for sentence in sentences1:
    #     print(sentence['content'],'   ',sentence['tokens'],'   ',sentence['ignored_tokens'],'\n')

    # print('\n\n')

    # for v in vocab['tokens']:
    #     print(v,'  ',vocab['tokens'][v]['text'],'   ',vocab['tokens'][v]['pos'],'   ',vocab['tokens'][v]['synsets'],'\n')
    
    # print('\n\n')

    # print(sent_similarity[0])
    # print('\n\n')
    # print(sent_similarity[1])

    # sent_similarity = similarities.sentences_similarity(sentences, 'list',sentences,'list',vocab)
    # # topics = extract_topic.extract_topic_of_sentences(sentences, vocab=vocab,source='list')
    # topics = extract_topic.extract_topic_of_sentences(sentences, similarity_matrix = sent_similarity)
    # print(time.time() - start,'  extracted topic of sentences')
    # for topic in topics:
    #     print('\n\n')
    #     for sentence in topic['sentences']:
    #         print(sentence['content'])


def test_sim_sentences_nouns_news():
    load_data.add_model_stopw(load_data.nlp)
    vocab = {'tokens':{},'ignored_tokens':{}}
    sentences = preprocess.extract_sentences(data_test,'list',vocab)
    nouns = preprocess.extract_nouns_from_news('oraciones calidad calidad comentario oraciones sentencia oración cama probando andar mesa puerta',vocab)
    similarities.create_similarity_matrix_words_wn(vocab)
    # score_sentences.sentence_most_popular(sentences,'list',vocab)
    # sent_similarity = similarities.sentences_similarity(sentences, 'list', nouns, 'news', vocab)
    topics = extract_topic.extract_topic_of_words(vocab,'list')
    topics = filters.filter_topics_for_news_similarity(0.5,vocab,topics,'list',nouns,unit='words')
    # score_topic.words_refers_to_news(nouns,topics,'list',vocab)
    # score_topic.words_most_popular(topics,'list',vocab)
    topics = extract_topic.map_sentences_into_topics(sentences, 'list',topics,'list',vocab)
    # score_topic.sentences_most_popular(topics,'list',vocab)
    # score_sentences.sentence_refers_to_news(sentences,'list',nouns, vocab)
    score_topic.sentences_refers_to_news(nouns, topics, 'list', vocab)
    print(topics)
    topics = filters.filter_topics_for_news_similarity(0.22,vocab,topics,'list',nouns)
    for topic in topics:
        print('EVALUATION ',evaluation.jensen_shannon_divergence(sentences,nouns))
    print(topics)

# test_preprocess()
# test_sim_words()
test_sim_sentences_nouns_news()
# test_sim_sentences()






"""test word similarity(passed) respect+"""
# s = [wn.synsets('caminar',pos='v',lang='spa'),wn.synsets('andar',pos='v',lang='spa')
#     ,wn.synsets('perro',pos='n',lang='spa'), wn.synsets('gato',pos='n',lang='spa'), wn.synsets('felino',pos='a',lang='spa')]
# for s1 in s:
#     for s2 in s:
#         if s1 == s2:
#             continue
#         max_sim = 0
#         for syn1 in s1:
#             for syn2 in s2:
#                 sim = wn.wup_similarity(syn1,syn2) or 0
#                 if sim > max_sim:
#                     max_sim = sim
#         print(s1,'  ',s2,' ',max_sim)
"""end test word similarity"""