import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import operator


st.title('Resumo de links do G1')
if 'resume_link' not in st.session_state:
    st.session_state.resume_link = 0
st.text_input("Informe o link", key="link")

method_chosed = st.radio(
   "What's your favorite method",
    ('tfidf', 'cosine'))
increment = st.button('Resumir')

def match_class(target):                                                        
    def do_match(tag):                                                          
        classes = tag.get('class', [])                                          
        return all(c in classes for c in target)                                
    return do_match 

def get_text_url(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html.parser')
    #remove marcações de scripts e style
    texto = soup.find_all(match_class(["content-text__container"]))
    all_text = ""
    for t in texto:
        all_text += t.get_text()
    return all_text

def pre_processamento_texto(corpus):
    corpus_alt = re.findall(r"\w+(?:'\w+)?|[^\w\s]", corpus)
    #lowcase
    corpus_alt = [t.lower() for t in corpus_alt]
    #remove stop words
    portugues_stops = stopwords.words('portuguese')
    corpus_alt = [t for t in corpus_alt if t not in portugues_stops]
    # #remove numbers
    corpus_alt = [re.sub(r'\d','', t) for t in corpus_alt]
    # #remove pontuation
    corpus_alt = [t for t in corpus_alt if t not in string.punctuation]
    # #remove accents
    # corpus_alt = [unidecode(t) for t in corpus_alt]

    return corpus_alt

def similaridade_tfidf(s1, s2):
    all_letter = s1 + s2
    vect = TfidfVectorizer()
    model_bow = vect.fit(all_letter)
    v1 = model_bow.transform([" ".join(s1)])
    v2 = model_bow.transform([" ".join(s2)])
    return cosine_similarity(v1, v2).reshape(-1)[0]

def similaridade(s1, s2):
    all_letter = s1 + s2
    vetor_bow = CountVectorizer()
    model_bow = vetor_bow.fit(all_letter)
    v1 = model_bow.transform([" ".join(s1)])
    v2 = model_bow.transform([" ".join(s2)])
    return cosine_similarity(v1, v2).reshape(-1)[0]

def matrix_similaridade(corpus, method='tfidf'):
    matriz = np.zeros((len(corpus), len(corpus)))
    for i in range(0, len(corpus)):
        for l in  range(0, len(corpus)):
            if(i != l):
                if(method == 'tfidf'):
                    matriz[i][l] =  similaridade_tfidf(corpus[i], corpus[l])
                else:    
                    matriz[i][l] =  similaridade(corpus[i], corpus[l])
    return matriz

def get_page_rank(texto, method='tfidf'):
    matrix = matrix_similaridade(texto, method)
    grafo = nx.from_numpy_array(matrix)
    return nx.pagerank_numpy(grafo)

if increment:
    text_news = get_text_url(st.session_state.link)
    tokens = sent_tokenize(text_news)
    corpus_processado = [pre_processamento_texto(t) for t in tokens]
    matrix = matrix_similaridade(corpus_processado)
    

    ranking = get_page_rank(corpus_processado, method=method_chosed)
    top5 = list(dict(sorted(ranking.items(), key = operator.itemgetter(1), reverse = True)).keys())[:5]
    texto_sumarizado = ""
    for t in top5:
        texto_sumarizado += tokens[t]
     
    st.write('Link = ',  st.session_state.link)  
    st.write('Resumo = ',  texto_sumarizado)  
