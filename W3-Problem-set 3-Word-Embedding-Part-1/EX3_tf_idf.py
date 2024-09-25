from EX2_term_document_matrix import create_term_document_matrix 
import pandas as pd 
import numpy as np 

def cal_tf(frequency: pd.Series) -> np.array:
    arr = frequency.to_numpy()
    arr = np.log10(1 + arr)
    return arr 

def cal_df(frequency: pd.Series) -> np.array:
    n = np.sum(frequency > 0, axis= 0)
    return n

def cal_idf(frequency: pd.Series) -> np.array:
    n = frequency.shape[0]
    df = cal_df(frequency)
    idf = np.log10(n/df)
    return idf

def cal_tf_idf(frequency: pd.Series) -> np.array:
    tf = cal_tf(frequency)
    idf = cal_idf(frequency)
    tf_idf = tf * idf
    return tf_idf

if __name__ == "__main__":
    data_path = "W1/data/all_data.csv"
    data = pd.read_csv(data_path)
    
    td_df = create_term_document_matrix(data)
    
    tfidf_df = td_df.apply(cal_tf_idf, axis=0)
    tfidf_df.to_csv('W1/result/TF_IDF.csv')