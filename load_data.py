import csv

def load_csv(data_file, delimiter=','):
    data = []
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            data.append({'content':row[3].lower(),'author':row[1],'date':row[2]})
    return data

def load_news(data_file):
    res = ""
    with open(data_file, 'r') as file:
        res = file.readline() 
    return res


def add_model_stopw(model, replace=True):
    with open('resources/es.txt', 'r') as file:
        stop_words = file.read().split('\n')
        if replace:
            new_stopw = set()
            for stop_word in stop_words:
                new_stopw.add(stop_word)
            model.Defaults.stop_words = new_stopw
        else:
            for stop_word in stop_words:
                model.Defaults.stop_words.add(stop_word)