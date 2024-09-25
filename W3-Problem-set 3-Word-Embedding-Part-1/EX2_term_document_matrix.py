import pandas as pd 
from collections import defaultdict 
import re 

def normalizer(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', ' ', s) 
    s = re.sub(r'\s+', ' ', s)
    return s
def split_text(text):
    text = normalizer(text)
    return text.split()

def create_vocabulary(data: pd.DataFrame) -> dict:
    vocabulary = set() 
    for idx, row in data.iterrows():
        word_list = split_text(row['text'])
        vocabulary.update(word_list)
    return sorted(list(vocabulary))

def get_articles(data: pd.DataFrame) -> dict:
    return data['title'].to_list()

def create_term_document_matrix(data: pd.DataFrame) -> pd.DataFrame:
    columns = get_articles(data)
    rows = create_vocabulary(data)
    df = pd.DataFrame(0, columns=columns, index=rows)
    
    for idx, row in data.iterrows():
        word_list = split_text(row['text'])
        for word in word_list:
            df.at[word, row['title']] += 1
    return df 

if __name__ == "__main__":
    data_path = "W1/data/all_data.csv"
    data = pd.read_csv(data_path)
    tdm = create_term_document_matrix(data)
    tdm.to_csv('W1/result/TD.csv')
    