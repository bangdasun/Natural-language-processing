'''
Data Structuring:
data['doc_type']['doc_id']['sent_id']['data_type']
Example:
data['weblog']['juancole.com_juancole_20051126063000_ENG_20051126_063000']['0001']['TOKENS']
'''

def load_data(path):
    fr = open(path,encoding='utf-8')
    data = dict()
    data_types = ['FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'HEAD']
    entry = None
    line = fr.readline()
    while (line):
        if line[0] == '#' and line[2] == 's':
            ids = line.split()[-1].split('-')
            entry = data
            for id in ids:	entry = entry.setdefault(id, dict())
            for t in data_types:	entry[t] = ['ROOT']
        elif line[0] == '#' and line[2] == 't':
            entry['TEXT'] = line[8:]
        elif line[0] != '#' and line[0] != '\n':
            token_data = line.split()
            entry['FORM'].append(token_data[1])
            entry['LEMMA'].append(token_data[2])
            entry['CPOSTAG'].append(token_data[3])
            entry['POSTAG'].append(token_data[4])
            entry['HEAD'].append(token_data[6])
        line = fr.readline()
    fr.close()
    return data
