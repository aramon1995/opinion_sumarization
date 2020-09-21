import similarities


def sentence_most_popular(sentences,source,vocab):
    similarity_matrix = similarities.sentences_similarity(sentences,source,sentences,source,vocab)
    for s in range(len(sentences)):
        if s in similarity_matrix[0][0]:
            sentences[s]['score'] = similarity_matrix[1][similarity_matrix[0][0].index(s)].mean()
        else:
            sentences[s]['score'] = 0
    

def sentence_refers_to_news(sentences, source, news, vocab):
    similarity_matrix = similarities.sentences_similarity(sentences, source, news, 'news', vocab)
    for i in range(len(sentences)):
        if i in similarity_matrix[0][0]:
            sentences[i]['score'] = similarity_matrix[1][similarity_matrix[0][0].index(i)][0]
        else:
            sentences[i]['score'] = 0
    

def heuristic_explanatory_ranking(topics):
    cwb = {}
    normB = 0
    k1 = 0.5
    b = 0.5
    for cluster in topics:
        cluster['words_counts'] = {}
        cluster['total_words'] = 0
        for sentence in cluster['sentences']:
            for w in sentence['tokens']:
                cluster['total_words'] += 1
                normB += 1
                if w not in cluster['words_counts'].keys():
                    cluster['words_counts'][w] = 1
                else:
                    cluster['words_counts'][w] += 1

                if w not in cwb.keys():
                    cwb[w] = 1
                else:
                    cwb[w] += 1

    for cluster in topics:
        for sentence in cluster['sentences']:
            score = 0
            for w in sentence['tokens']:
                idf = (normB - cwb[w] + 0.5)/(cwb[w] + 0.5)
                score += (idf)*((cluster['words_counts'][w]*(k1 + 1))/(cluster['words_counts'][w] + k1))
            
            sentence['score'] = score
        del cluster['words_counts']
        del cluster['total_words']
