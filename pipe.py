import load_data
import preprocess
import similarities
import extract_topic
import score_topic
import score_sentences
import polarity
import filters
import evaluation



def v1(data_dir,source,news_dir):
    load_data.add_model_stopw(load_data.nlp)
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
    f = open('result/'+source+'.txt','w')
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





def v2(data_dir,source,news_dir):
    load_data.add_model_stopw(load_data.nlp)
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
    f = open('result/v2/'+source+'.txt','w')
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






def v3_4(data_dir,source,news_dir):
    load_data.add_model_stopw(load_data.nlp)
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
    f = open('result/v3/'+source+'.txt','w')
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
    f = open('result/v4/'+source+'.txt','w')
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




def v5_6(data_dir,source,news_dir):
    # load_data.add_model_stopw(load_data.nlp)
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
    f = open('result/v5/'+source+'.txt','w')
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
    f = open('result/v6/'+source+'.txt','w')
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




def v1_3_4_we(data_dir,source,news_dir):
    load_data.add_model_stopw(load_data.nlp)
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
    f = open('result/v1_we/'+source+'.txt','w')
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

    f = open('result/v3_we/'+source+'.txt','w')
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
    f = open('result/v4_we/'+source+'.txt','w')
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





def v2_5_6_we(data_dir,source,news_dir):
    load_data.add_model_stopw(load_data.nlp)
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
    f = open('result/v2_we/'+source+'.txt','w')
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

    f = open('result/v5_we/'+source+'.txt','w')
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
    f = open('result/v6_we/'+source+'.txt','w')
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
    from os.path import join
    from os import listdir
    funct_map = {'v1':v1,'v2':v2,'v3_4':v3_4,'v5_6':v5_6,'v1_3_4_we':v1_3_4_we,'v2_5_6_we':v2_5_6_we}
    start = time.time()
    opinions_path = 'data/etecsa/opinions/'
    news_path = 'data/etecsa/news/'
    files = listdir(opinions_path)
    files.sort()
    print(files)
    for f in files:
        funct_map[sys.argv[1]](join(opinions_path,f),f.split('.')[0],join(news_path,f.split('.')[0]+'.txt'))    
    
    # v1(join(opinions_path,'prueba_4g.csv'),'pruebas_4g',join(news_path,'prueba_4g.txt'))  
    print('time ', time.time()-start)


    import sounddevice as sd
    import soundfile as sf

    filename = '/media/Data/Storage/file_example_WAV_1MG.wav'
    # Extract data and sampling rate from file
    data, fs = sf.read(filename, dtype='float32')  
    sd.play(data, fs)
    status = sd.wait()
