import load_data
import similarities
import extract_topic
import score_topic
import score_sentences
import polarity
import filters
import evaluation
import os



def v1(data_dir,source,news_dir,out):
    opinions = load_data.load_csv(data_dir)
    data = [o['content'] for o in opinions]
    news = load_data.load_news(news_dir)
    vocab = {'tokens':{},'ignored_tokens':{}}
    sentences = preprocess.extract_sentences(data,source,vocab)
    nouns = preprocess.extract_nouns_from_news(news,vocab)
    polarity.sentence_polarity(sentences,vocab)
    sentences = filters.filter_sentences_for_polarity(sentences)
    similarities.create_similarity_matrix_words_wn(vocab)
    topics_words,sil = extract_topic.extract_topic_of_words(vocab,source)
    topics_words = filters.filter_topics_for_news_similarity(0.5, vocab, topics_words, source, nouns, unit='words')
    topics = extract_topic.map_sentences_into_topics(sentences,source,topics_words,source,vocab)
    score_sentences.heuristic_explanatory_ranking(topics)
    out_path = out+'v1/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        print('topic: ',topics_words[topic['topic_index']]['tokens'],'\n\n')
        f.write('topic: '+str(topics_words[topic['topic_index']]['tokens'])+'\n\n')
        sents = topic['sentences']
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        if cant == 0:
            print('NO SENTENCES FOR TOPIC: ', topic['topic_index'])
            f.write('NO SENTENCES FOR TOPIC: '+str(topic['topic_index'])+'\n')
            continue
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()





def v2(data_dir,source,news_dir,out):
    opinions = load_data.load_csv(data_dir)
    data = [o['content'] for o in opinions]
    news = load_data.load_news(news_dir)
    vocab = {'tokens':{},'ignored_tokens':{}}
    sentences = preprocess.extract_sentences(data,source,vocab)
    nouns = preprocess.extract_nouns_from_news(news,vocab)
    polarity.sentence_polarity(sentences,vocab)
    sentences = filters.filter_sentences_for_polarity(sentences)
    similarities.create_similarity_matrix_words_wn(vocab)
    
    topics, sil = extract_topic.extract_topic_of_sentences(sentences,source,vocab=vocab)
    topics = filters.filter_topics_for_news_similarity(0.5,vocab,topics,source,nouns)
    score_sentences.heuristic_explanatory_ranking(topics)
    out_path = out+'v2/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        sents = topic['sentences']
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()






def v3_4(data_dir,source,news_dir,out):
    opinions = load_data.load_csv(data_dir)
    data = [o['content'] for o in opinions]
    news = load_data.load_news(news_dir)
    vocab = {'tokens':{},'ignored_tokens':{}}
    sentences = preprocess.extract_sentences(data,source,vocab)
    nouns = preprocess.extract_nouns_from_news(news,vocab)
    polarity.sentence_polarity(sentences,vocab)
    sentences = filters.filter_sentences_for_polarity(sentences)
    similarities.create_similarity_matrix_words_wn(vocab)
    topics_words,sil = extract_topic.extract_topic_of_words(vocab,source)
    topics_words = filters.filter_topics_for_news_similarity(0.5, vocab, topics_words, source, nouns, unit='words')
    topics = extract_topic.map_sentences_into_topics(sentences,source,topics_words,source,vocab)
    out_path = out+'v3/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        print('topic: ',topics_words[topic['topic_index']]['tokens'],'\n\n')
        f.write('topic: '+str(topics_words[topic['topic_index']]['tokens'])+'\n\n')
        sents = topic['sentences']
        score_sentences.sentence_most_popular(sents,source,vocab)
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        if cant == 0:
            print('NO SENTENCES FOR TOPIC: ', topic['topic_index'])
            f.write('NO SENTENCES FOR TOPIC: '+str(topic['topic_index'])+'\n')
            continue
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()
    score_sentences.sentence_refers_to_news(sentences,source,nouns,vocab)
    out_path = out+'v4/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        print('topic: ',topics_words[topic['topic_index']]['tokens'],'\n\n')
        f.write('topic: '+str(topics_words[topic['topic_index']]['tokens'])+'\n\n')
        sents = topic['sentences']
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        if cant == 0:
            print('NO SENTENCES FOR TOPIC: ', topic['topic_index'])
            f.write('NO SENTENCES FOR TOPIC: '+str(topic['topic_index'])+'\n')
            continue
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()




def v5_6(data_dir,source,news_dir,out):    
    opinions = load_data.load_csv(data_dir)
    data = [o['content'] for o in opinions]
    news = load_data.load_news(news_dir)
    vocab = {'tokens':{},'ignored_tokens':{}}
    sentences = preprocess.extract_sentences(data,source,vocab)
    nouns = preprocess.extract_nouns_from_news(news,vocab)
    polarity.sentence_polarity(sentences,vocab)
    sentences = filters.filter_sentences_for_polarity(sentences)
    similarities.create_similarity_matrix_words_wn(vocab)
    
    topics, sil = extract_topic.extract_topic_of_sentences(sentences,source,vocab=vocab)
    topics = filters.filter_topics_for_news_similarity(0.5,vocab,topics,source,nouns)
    out_path = out+'v5/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        sents = topic['sentences']
        score_sentences.sentence_most_popular(sents,source,vocab)
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()

    score_sentences.sentence_refers_to_news(sentences,source,nouns,vocab)
    out_path = out+'v6/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        sents = topic['sentences']        
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()




def v1_3_4_we(data_dir,source,news_dir,out):
    opinions = load_data.load_csv(data_dir)
    data = [o['content'] for o in opinions]
    news = load_data.load_news(news_dir)
    vocab = {'tokens':{},'ignored_tokens':{}}
    sentences = preprocess.extract_sentences(data,source,vocab)
    nouns = preprocess.extract_nouns_from_news(news,vocab)
    polarity.sentence_polarity(sentences,vocab)
    sentences = filters.filter_sentences_for_polarity(sentences)
    similarities.create_similarity_matrix_words_we(vocab)
    topics_words,sil = extract_topic.extract_topic_of_words(vocab,source)
    topics_words = filters.filter_topics_for_news_similarity(0.5, vocab, topics_words, source, nouns, unit='words')
    topics = extract_topic.map_sentences_into_topics(sentences,source,topics_words,source,vocab)
    
    score_sentences.heuristic_explanatory_ranking(topics)
    out_path = out+'v1_we/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        print('topic: ',topics_words[topic['topic_index']]['tokens'],'\n\n')
        f.write('topic: '+str(topics_words[topic['topic_index']]['tokens'])+'\n\n')
        sents = topic['sentences']
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        if cant == 0:
            print('NO SENTENCES FOR TOPIC: ', topic['topic_index'])
            f.write('NO SENTENCES FOR TOPIC: '+str(topic['topic_index'])+'\n')
            continue
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()

    out_path = out+'v3_we/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        print('topic: ',topics_words[topic['topic_index']]['tokens'],'\n\n')
        f.write('topic: '+str(topics_words[topic['topic_index']]['tokens'])+'\n\n')
        sents = topic['sentences']
        score_sentences.sentence_most_popular(sents,source,vocab)
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        if cant == 0:
            print('NO SENTENCES FOR TOPIC: ', topic['topic_index'])
            f.write('NO SENTENCES FOR TOPIC: '+str(topic['topic_index'])+'\n')
            continue
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()

    score_sentences.sentence_refers_to_news(sentences,source,nouns,vocab)
    out_path = out+'v4_we/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        print('topic: ',topics_words[topic['topic_index']]['tokens'],'\n\n')
        f.write('topic: '+str(topics_words[topic['topic_index']]['tokens'])+'\n\n')
        sents = topic['sentences']
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        if cant == 0:
            print('NO SENTENCES FOR TOPIC: ', topic['topic_index'])
            f.write('NO SENTENCES FOR TOPIC: '+str(topic['topic_index'])+'\n')
            continue
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()





def v2_5_6_we(data_dir,source,news_dir,out):
    opinions = load_data.load_csv(data_dir)
    data = [o['content'] for o in opinions]
    news = load_data.load_news(news_dir)
    vocab = {'tokens':{},'ignored_tokens':{}}
    sentences = preprocess.extract_sentences(data,source,vocab)
    nouns = preprocess.extract_nouns_from_news(news,vocab)
    polarity.sentence_polarity(sentences,vocab)
    sentences = filters.filter_sentences_for_polarity(sentences)
    similarities.create_similarity_matrix_words_we(vocab)
    
    topics, sil = extract_topic.extract_topic_of_sentences(sentences,source,vocab=vocab)
    topics = filters.filter_topics_for_news_similarity(0.5,vocab,topics,source,nouns)
    
    score_sentences.heuristic_explanatory_ranking(topics)
    out_path = out+'v2_we/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        sents = topic['sentences']
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()

    out_path = out+'v5_we/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        sents = topic['sentences']
        score_sentences.sentence_most_popular(sents,source,vocab)
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()

    score_sentences.sentence_refers_to_news(sentences,source,nouns,vocab)
    out_path = out+'v6_we/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(out_path+source+'.txt','w')
    print('silhouette: ',sil,'\n\n')
    f.write('silhouette: '+str(sil)+'\n\n')
    for topic in topics:
        sents = topic['sentences']
        sents.sort(key=lambda s:s['score'],reverse = True)
        cant = 3 if len(sents)>3 else len(sents)
        s = []
        for sentence in range(cant):
            s.append(sents[sentence])
            print(sents[sentence]['content'])
            f.write(str(sents[sentence]['content'])+'\n')
            if sentence<cant:
                print('*********')
                f.write('*********\n')
            print('\n')
        js_topic = evaluation.jensen_shannon_divergence(sents,s)
        js_opinions = evaluation.jensen_shannon_divergence(sentences,s)
        js_news = evaluation.jensen_shannon_divergence(nouns,s)
        print('jensen shannon divergence respect to topic: ',js_topic)
        print('jensen shannon divergence respect to all opinions: ',js_opinions)
        print('jensen shannon divergence respect to news: ',js_news)
        f.write('jensen shannon divergence respect to topic: '+str(js_topic)+'\n')
        f.write('jensen shannon divergence respect to all opinions: '+str(js_opinions)+'\n')
        f.write('jensen shannon divergence respect to news: '+str(js_news)+'\n')
        print('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
        f.write('\n\n***----------------------------------------------------------------------------------------------------***\n\n')
    f.close()

if __name__ == "__main__":
    import sys
    import time
    import multiprocessing as mp

    function_map = {'v1':v1,'v2':v2,'v3_4':v3_4,'v5_6':v5_6,'v1_3_4_we':v1_3_4_we,'v2_5_6_we':v2_5_6_we}
    function = sys.argv[1]
    opinion_path = sys.argv[2]
    news_path = sys.argv[3]
    out_path = sys.argv[4]
    preprocess = sys.argv[5]
    if preprocess == 'spacy':
        import preprocess
        load_data.add_model_stopw(preprocess.model)
    elif preprocess == 'nltk':
        import preprocess_NLTK as preprocess
    else:
        raise Exception('preprocess option not valid, valid options are \'nltk\' or \'spacy\'')
    if function not in function_map:
        raise Exception('function not valid, posible function: '+function_map.keys())
    if not os.path.exists(opinion_path):
        raise Exception('opinion path: '+opinion_path+' is not valid')
    if not os.path.exists(news_path):
        raise Exception('news path: '+news_path+' is not valid')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    start = time.time()
    if os.path.isfile(opinion_path):
        source = opinion_path.split('/')[-1].split('.')[0]
        news = os.path.join(news_path,source)+'.txt'
        function_map[function](opinion_path,source,news,out_path)
    else:
        files = os.listdir(opinion_path)
        files.sort()
        
        # for f in files:
        #     opinion = os.path.join(opinion_path,f)
        #     source = f.split('.')[0]
        #     news = os.path.join(news_path,source)+'.txt'
        #     function_map[function](opinion,source,news,out_path)
        
        params = []
        for f in files:
            opinion = os.path.join(opinion_path,f)
            source = f.split('.')[0]
            news = os.path.join(news_path,source)+'.txt'
            params.append((opinion,source,news,out_path))
        pool = mp.Pool(mp.cpu_count())
        pool.starmap(function_map[function],params)
        pool.close()
        pool.join()
    print('EXECUTION TIME: ',time.time()-start)