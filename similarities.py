import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity


def create_similarity_matrix_words_wn(vocab):
    words_in_sim_matrix = []
    t = list(vocab['tokens'].items())
    cant_tokens = len(t)
    similarity_matrix = np.ones((cant_tokens,cant_tokens))
    actual_t1 = -1
    for t1_index in range(cant_tokens - 1):
        t1 = t[t1_index]
        if len(t1[1]['synsets']) == 0:
            continue
        actual_t1 += 1
        words_in_sim_matrix.append(t1[0])
        actual_t2 = 0
        for t2_index in range(t1_index + 1, cant_tokens):
            t2 = t[t2_index]
            if len(t2[1]['synsets']) == 0:
                continue
            actual_t2 += 1
            # max_similarity = 0
            # for t1_syn in t1[1]['synsets']:
            #     for t2_syn in t2[1]['synsets']:
            #         similarity = wn.wup_similarity(t1_syn,t2_syn) or 0
            #         if similarity > max_similarity:
            #             max_similarity = similarity
            # similarity_matrix[actual_t1,actual_t1 + actual_t2] = similarity_matrix[actual_t1 + actual_t2,actual_t1] = max_similarity
            similarity_matrix[actual_t1,actual_t1 + actual_t2] = similarity_matrix[actual_t1 + actual_t2,actual_t1] = wn.wup_similarity(t1[1]['synsets'][0],t2[1]['synsets'][0]) or 0

    if len(t2[1]['synsets']) != 0:
        words_in_sim_matrix.append(t2[0])

    similarity_matrix = similarity_matrix[:len(words_in_sim_matrix),:len(words_in_sim_matrix)]
    vocab['similarity_matrix_of_words'] = similarity_matrix
    vocab['words_in_similarity_matrix'] = words_in_sim_matrix



def create_similarity_matrix_words_we(vocab):
    words_in_sim_matrix = []
    vectors = []
    cant_tokens = len(vocab['tokens'].keys())
    for t1_index in range(cant_tokens):
        t1 = list(vocab['tokens'].items())[t1_index]
        if len(t1[1]['vector']) == 0:
            continue
        vectors.append(t1[1]['vector'])
        words_in_sim_matrix.append(t1[0])

    vector_matrix = np.array(vectors)
    vector_matrix = (vector_matrix - vector_matrix.min())/(vector_matrix.max() - vector_matrix.min())
    similarity_matrix = cosine_similarity(vector_matrix)
    np.fill_diagonal(similarity_matrix,1)
    similarity_matrix = np.round(similarity_matrix,decimals=7)
    similarity_matrix = np.where(similarity_matrix<1,similarity_matrix,1)
    vocab['similarity_matrix_of_words'] = similarity_matrix
    vocab['words_in_similarity_matrix'] = words_in_sim_matrix





def sentences_similarity1(sentence_list1, source1, sentence_list2, source2, vocab):
    sentences_list1_in_similarity_matrix = []
    sentences_list2_in_similarity_matrix = []
    similarity_matrix_of_sentences = np.ones((len(sentence_list1),len(sentence_list2)))
    actual_s1 = 0
    for sentence1_index in range(len(sentence_list1)):
        idf_of_words_in_sentence1 = np.empty(len(sentence_list1[sentence1_index]['tokens']))
        actual_s2 = 0
        for sentence2_index in range(len(sentence_list2)):
            idf_of_words_in_sentence2 = np.empty(len(sentence_list2[sentence2_index]['tokens']))
            similarity_matrix_of_sentence1_sentence2 = np.zeros((len(sentence_list1[sentence1_index]['tokens']),len(sentence_list2[sentence2_index]['tokens'])))
            actual_t1 = 0
            for t1 in sentence_list1[sentence1_index]['tokens']:
                try:
                    index_of_t1_in_vocab = vocab['words_in_similarity_matrix'].index(t1)
                except:
                    index_of_t1_in_vocab = None
                if index_of_t1_in_vocab == None:
                    continue
                idf_of_words_in_sentence1[actual_t1] = len(vocab['tokens'][t1]['hit_'+source1])/len(sentence_list1)
                actual_t2 = 0
                for t2 in sentence_list2[sentence2_index]['tokens']:
                    try:
                        index_of_t2_in_vocab = vocab['words_in_similarity_matrix'].index(t2)
                    except:
                        index_of_t2_in_vocab = None
                    if index_of_t2_in_vocab == None:
                        continue
                    idf_of_words_in_sentence2[actual_t2] = len(vocab['tokens'][t2]['hit_'+source2])/len(sentence_list2)
                    similarity_matrix_of_sentence1_sentence2[actual_t1,actual_t2] = vocab['similarity_matrix_of_words'][index_of_t1_in_vocab,index_of_t2_in_vocab]
                    actual_t2 += 1
                actual_t1 += 1
            if actual_t1 == 0 or actual_t2 == 0:
                continue
            similarity_matrix_of_sentence1_sentence2 = similarity_matrix_of_sentence1_sentence2[:actual_t1,:actual_t2]
            idf_of_words_in_sentence1 = idf_of_words_in_sentence1[:actual_t1]
            idf_of_words_in_sentence2 = idf_of_words_in_sentence2[:actual_t2]
            if actual_s1 == 0:
                sentences_list2_in_similarity_matrix.append(sentence2_index)
            w1_t2 = (similarity_matrix_of_sentence1_sentence2.max(axis=1)*idf_of_words_in_sentence1).sum()/idf_of_words_in_sentence1.sum()
            w2_t1 = (similarity_matrix_of_sentence1_sentence2.max(axis=0)*idf_of_words_in_sentence2).sum()/idf_of_words_in_sentence2.sum()
            similarity_matrix_of_sentences[actual_s1,actual_s2] = (w1_t2 + w2_t1)/2
            actual_s2 += 1
        if actual_t1 == 0 or actual_t2 == 0:
            continue
        actual_s1 += 1
        sentences_list1_in_similarity_matrix.append(sentence1_index)
    similarity_matrix_of_sentences = similarity_matrix_of_sentences[:len(sentences_list1_in_similarity_matrix),:len(sentences_list2_in_similarity_matrix)]
    return [[sentences_list1_in_similarity_matrix,sentences_list2_in_similarity_matrix],similarity_matrix_of_sentences]




def sentences_similarity(sentence_list1, source1, sentence_list2, source2, vocab):
    similarity_matrix = []
    sentences_list1_in_similarity_matrix = []
    sentences_list2_in_similarity_matrix = []

    for sentence_index1 in enumerate(sentence_list1):
        similarity_row = []
        for sentence_index2 in enumerate(sentence_list2):
            tokens_similarity = []
            idf1 = []
            for token1 in sentence_index1[1]['tokens']:
                if token1 not in vocab['words_in_similarity_matrix']:
                    continue
                idf1.append(len(sentence_list1)/len(vocab['tokens'][token1]['hit_'+source1]))
                idf2 = []
                tokens_row = []
                for token2 in sentence_index2[1]['tokens']:
                    if token2 not in vocab['words_in_similarity_matrix']:
                        continue
                    idf2.append(len(sentence_list2)/len(vocab['tokens'][token2]['hit_'+source2]))
                    tokens_row.append(vocab['similarity_matrix_of_words'][vocab['words_in_similarity_matrix'].index(token1),vocab['words_in_similarity_matrix'].index(token2)])
                if len(tokens_row) != 0:
                    tokens_similarity.append(tokens_row)

            if len(tokens_similarity) == 0:
                continue
            if sentence_index2[0] not in sentences_list2_in_similarity_matrix:
                sentences_list2_in_similarity_matrix.append(sentence_index2[0])
            tokens_similarity = np.array(tokens_similarity)
            idf1 = np.array(idf1)
            idf2 = np.array(idf2)
            s1 = (tokens_similarity.max(axis = 1)*idf1).sum()/idf1.sum()
            s2 = (tokens_similarity.max(axis = 0)*idf2).sum()/idf2.sum()
            similarity_row.append((s1+s2)/2)
        if len(similarity_row) == 0:
            continue
        similarity_matrix.append(similarity_row)
        sentences_list1_in_similarity_matrix.append(sentence_index1[0])
    return [[sentences_list1_in_similarity_matrix,sentences_list2_in_similarity_matrix],np.array(similarity_matrix)]