import spacy
from collections import Counter, defaultdict
import pickle
from utils import get_paragraphs_as_text
from multiprocessing import Pool, Process
import time

def extract_keys(paragraphs, n = 20): 
    keys_tags = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "dobj", "obj"]
    parser = spacy.load('en')
    words, words_with_indices = [], []
    
    for paragraph in paragraphs:
        doc = parser(paragraph)
        for i, tok in enumerate(doc):
            if tok.dep_ in keys_tags:
                words.append(str(tok).lower())
                words_with_indices.append((str(tok).lower(), i))
                
    cnt = Counter(words)
    most_common_words = [word for word, rep in cnt.most_common(n)]
    word2indices = defaultdict(lambda : [])
    
    for word, i in words_with_indices:
        if word in most_common_words:
            word2indices[word].append(i)
    
    return dict(word2indices)

if __name__ == '__main__':
    s = time.time()
    
    print(time.time())
    
    paragraph_pairs = get_paragraphs_as_text(max_len = 450, min_len = 20, tags = [[0,1,2]])
    
    with open('paragraph_pairs_text.pkl', 'wb') as pkl:
        pickle.dump(paragraph_pairs, pkl)
        
    print(time.time())
    
    agents = 16
    chunksize = 3
    with Pool(processes = agents) as pool:
        keys = pool.map(extract_keys, paragraph_pairs, chunksize)
    
    with open('paragraph_pairs_keys.pkl', 'wb') as pkl:
        pickle.dump(keys, pkl)
        
    print(time.time() - s)