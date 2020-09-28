import similarities
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import numpy as np

def extract_topic_of_words(vocab, source):
    words_in_similarity_matrix = []
    distance_matrix = []
    for t1_index in range(len(vocab['words_in_similarity_matrix'])):
        if 'hit_'+source not in vocab['tokens'][vocab['words_in_similarity_matrix'][t1_index]].keys():
            continue
        words_in_similarity_matrix.append(vocab['words_in_similarity_matrix'][t1_index])
        row = []
        for t2_index in range(len(vocab['words_in_similarity_matrix'])):
            if 'hit_'+source not in vocab['tokens'][vocab['words_in_similarity_matrix'][t2_index]].keys():
                continue
            row.append(1 - vocab['similarity_matrix_of_words'][t1_index,t2_index])
        distance_matrix.append(row)
    distance_matrix = np.array(distance_matrix)
    HAC = AgglomerativeClustering(affinity='precomputed', compute_full_tree=True, linkage='average', n_clusters=None, distance_threshold=(distance_matrix.min(axis=1).mean()+distance_matrix.max(axis=1).mean())/2)
    # HAC = AgglomerativeClustering(affinity='precomputed', compute_full_tree=True, linkage='average', n_clusters=None, distance_threshold=((distance_matrix.mean()+distance_matrix.min(axis=1).mean())/2))
    HAC.fit(distance_matrix)
    if HAC.n_clusters_ == len(vocab['words_in_similarity_matrix']) or HAC.n_clusters_ == 1:
        sil = 0
    else:
        sil = metrics.silhouette_score(distance_matrix, HAC.labels_,metric = 'precomputed')
    clusters = [{'tokens':[]} for _ in range(HAC.n_clusters_)]
    for i in range(len(HAC.labels_)):
        clusters[HAC.labels_[i]]['tokens'].append(words_in_similarity_matrix[i])
        vocab['tokens'][words_in_similarity_matrix[i]]['hit_topics_of_words_from_'+source] = [HAC.labels_[i]]
    return clusters,sil



def extract_topic_of_sentences(sentences, source, vocab = None, similarity_matrix = []):
    if len(similarity_matrix) == []:
        if vocab == None or source == None:
            raise Exception('If similarity matrix is None, vocab and source must be provided')
        else:
            similarity_matrix = similarities.sentences_similarity(sentences, source, sentences, source, vocab)
    distance_matrix = 1 - similarity_matrix[1]
    HAC = AgglomerativeClustering(affinity='precomputed', compute_full_tree=True, linkage='average', n_clusters=None, distance_threshold=(distance_matrix.min(axis=1).mean()+distance_matrix.max(axis=1).mean())/2)
    print(similarity_matrix)
    HAC.fit(distance_matrix)
    if HAC.n_clusters_ == 1 or HAC.n_clusters_ == len(distance_matrix[0]):
        sil = 0
    else:
        sil = metrics.silhouette_score(distance_matrix, HAC.labels_,metric = 'precomputed')
    clusters = [{'sentences':[]} for _ in range(HAC.n_clusters_)]
    for i in range(len(HAC.labels_)):
        clusters[HAC.labels_[i]]['sentences'].append(sentences[similarity_matrix[0][0][i]])
        for t in sentences[similarity_matrix[0][0][i]]['tokens']:
            if 'hit_topics_of_sentences_from_'+source in vocab['tokens'][t].keys():
                vocab['tokens'][t]['hit_topics_of_sentences_from_'+source].append(HAC.labels_[i])
            else:
                vocab['tokens'][t]['hit_topics_of_sentences_from_'+source] = [HAC.labels_[i]]
    return clusters, sil



def map_sentences_into_topics(sentences, source_sentences, topics, source_topics, vocab):
    similarity_topics_sentences = similarities.sentences_similarity(sentences, source_sentences, topics, 'topics_of_words_from_'+source_topics, vocab)
    labels = similarity_topics_sentences[1].argmax(axis = 1)
    clusters = [{'sentences':[], 'topic_index':i} for i in range(max(set(labels))+1)]
    for i in range(len(labels)):
        clusters[labels[i]]['sentences'].append(sentences[similarity_topics_sentences[0][0][i]])    
    clusters = list(filter(lambda x: len(x['sentences']) != 0,clusters))
    for cluster_index in range(len(clusters)):
        for sentence in clusters[cluster_index]['sentences']:
            for t in sentence['tokens']:
                if 'hit_topics_of_sentences_from_'+source_sentences in vocab['tokens'][t].keys():
                    vocab['tokens'][t]['hit_topics_of_sentences_from_'+source_sentences].append(cluster_index)
                else:
                    vocab['tokens'][t]['hit_topics_of_sentences_from_'+source_sentences] = [cluster_index]
    return clusters