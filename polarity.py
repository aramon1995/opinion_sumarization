def load_es_swn(swn_dir):
    res = {}
    try:
        with open(swn_dir, 'r') as file:
            for line in file.readlines():
                line = line.replace("\n", '')
                data = line.split("\t")
                all_info = str(data[1]).split(' ')
                res[data[0]] = (float(all_info[1]), float(all_info[2]))
    except Exception as err:
        raise Exception("Cannot load es SWN due: " + str(err))
    return res


def sentence_polarity(sentences, vocab, swn_dir = 'resources/swn_es_1.0.txt'):
    swn = load_es_swn(swn_dir)
    for sentence in sentences:
        sentence['polarity'] = [0,0]
        for token in sentence['tokens']:
            pos_val = neg_val = None
            try:
                pos_val, neg_val = swn[vocab['tokens'][token]['text']]
                if pos_val is None or neg_val is None:
                    pos_val, neg_val = swn[token]
            except:
                pass
            if pos_val is None or neg_val is None:
                pos_val = neg_val = 0
            sentence['polarity'][0] += pos_val
            sentence['polarity'][1] += neg_val
        for token in sentence['ignored_tokens']:
            pos_val = neg_val = None
            try:
                pos_val, neg_val = swn[vocab['ignored_tokens'][token]['text']]
                if pos_val is None or neg_val is None:
                    pos_val, neg_val = swn[token]
            except:
                pass
            if pos_val is None or neg_val is None:
                pos_val = neg_val = 0
            sentence['polarity'][0] += pos_val
            sentence['polarity'][1] += neg_val
