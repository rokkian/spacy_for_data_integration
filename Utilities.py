import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandasql as pds
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
from spacy import displacy
from copy import deepcopy
# import networkx as nx
from pulp import *      # dovrebbe essere una libreria per l'ottimizzazione lineare
import seaborn as sns; sns.set_theme()

#funzione per trasformare il DataFrame in un dict di documenti
def dataframeInDocs(df, verbose=False):
    docs = {}
    for col in df.columns:
        docs.update({col : ' '.join(list(df[col]))})
    if verbose:
        print(f'Creato dict di documenti con {len(docs.keys())} attributi: {list(docs.keys())}')
    return docs

#funzione per campionare i record che non hanno nan
def sampleDF_clean(df:pd.DataFrame, size=100):
    #Righe che non hanno elementi 'nan' in alcuna colonna
    if size >= df.shape[0]:
        return df
    df = df[(df != 'nan').all(axis=1)]
    df = df[(df != 'NaN').all(axis=1)]
    df = df[(df != ' ').all(axis=1)]
    df = df[(df != '').all(axis=1)]
    if size >= df.shape[0]:
        return df
    assert df.shape[0] > 0
    sample = df.sample(size)
    return sample

#Funzione di calcolo della ricchezza semantica
def lexicalRichness(doc: spacy.tokens.doc.Doc):
    return len(set([w.text for w in doc]))/len([w.text for w in doc])

def preprocessingNLTK(text):
    #cerca di rimuovere le stop words
    tokens = word_tokenize(text)
    cell_tokens_wo_sw = [word for word in tokens if word not in stopwords.words()]
    return ' '.join(cell_tokens_wo_sw)

#----------------------------Classi ------------------------------------------------------------

#Classe del singolo documento
class Doc:
    name = ""
    text = ""
    semanticRichness = 0

    def __init__(self, name, text, n_celle, nlp, n_token_sep, pre=False):
        self.name = name
        self.text = text
        self.n_celle = n_celle
        self.n_token_separator = n_token_sep
        self.prep = pre
        self.NLP = nlp
        #aggiunta elaborazione del testo (stop words e normalizzazione lemmi e stem)(test se migliora le performance)
        if pre:
            self.text_clean = preprocessingNLTK(self.text)
            self.nlp = nlp(self.text_clean)
            self.reduction_preprocessing = len(self.text_clean)/len(self.text)
        else:
            self.nlp = nlp(self.text)
        self.numero_parole_diverse_documento = len(set([w.text for w in self.nlp]))
        self.numero_parole_documento = len([w.text for w in self.nlp])
        self.lexicalRichness = self.numero_parole_diverse_documento/(self.numero_parole_documento-(self.n_token_separator*n_celle-1)) #corretta togliendo il numero di token separatori dal testo
        self.media_token_per_cella = (self.numero_parole_documento - (self.n_token_separator*n_celle))/self.n_celle #la media è corretta togliendo il numero di token separatori nel testo

    def doNLP(self, model=spacy.load("en_core_web_sm")):
        if self.prep:
            self.nlp = model(self.text_clean)
        else:
            self.nlp = model(self.text)

    def describeDoc(self):
        print(f'Documento {self.name}\ncon in origine {self.n_celle} celle, con una media di {self.media_token_per_cella} token per cella')
        print(f'Ricchezza lessicale del testo {self.lexicalRichness}')
        if self.prep:
            print(f'Preprocessing con riduzione al {self.reduction_preprocessing}%')

#Classe di uno schema di documenti 
#       lista dei modelli: (en_core_web_sm('small'), en_core_web_md('medium'), en_core_web_lg('large')), vectors(en_vectors_web_lg)
class SchemaDoc:

    def __init__(self, df, sep=' ', nlp=spacy.load("en_core_web_sm"), pre=False, verbose=False):
        nlp.max_length = 1e7        #imposta la massima dimensione del singolo testo
        n_token_sep = len([w.text for w in nlp(sep)])
        self.NLP_ = nlp

        #if type(df) is pd.DataFrame or type(df) is pandas.core.frame.DataFrame:
        self.docs_ = list(df.columns)
        for doc in self.docs_:
            if verbose:
                print(f'Creation doc {doc}')
            setattr(self, doc, Doc(doc, sep.join(list(df[doc])), len(df[doc]), nlp, n_token_sep, pre=pre))
    
        if verbose:
            print('Fine SchemaDoc')

        #else:
        #    raise TypeError(f"Per creare uno SchemaDoc serve un DataFrame, non un {type(df)}")

    def getListDocumenti(self):
        return self.docs_

    def doNLPs(self, model=spacy.load("en_core_web_sm"), verbose=False):
        """Metodo da utilizzare per applicare il modello a tutti i Doc"""
        for doc in self.docs_:
            if verbose:
                print(f"Doing NLP of {doc}")
            getattr(self, doc).doNLP(model)
        if verbose:
            print("Fine NLPs")

    def getDictDocumenti(self):
        d = {}
        for doc in self.docs_:
            d.update({doc : getattr(self, doc)})
        return d

    def describeDocs(self):
        print("Description of the documents of the schema:")
        for doc in self.docs_:
            print(f'{getattr(self, doc).name}:\t text length: {len(getattr(self, doc).text)} Bytes;\t lexical richness: {getattr(self, doc).lexicalRichness}')

    def __repr__(self):
        return f'{self.docs_}'


#---------------------------------Tabelle Similarità----------------------------------------------

#def SimilarityFunction(x):
#    return lev.get_sim_score(preprocess_s(x['A']), preprocess_s(x['B']))
    #return lev.get_sim_score(preprocess_s(x['A']), preprocess_s(x['B'])

def SimilarityTable(A:SchemaDoc,B:SchemaDoc):
    DA=pd.DataFrame({'A': A.docs_})
    DB=pd.DataFrame({'B': B.docs_})
    PCC = DA.assign(key=1).merge(DB.assign(key=1), on='key').drop('key', 1)
    PCC.columns=['A','B']
    #PCC['sim']=PCC.apply(SimilarityFunction, axis=1)                   
    PCC['sim']=0         
    return PCC

def toSimMatrix(SimTable):
    A_label = SimTable.columns[0]
    B_label = SimTable.columns[1]
    A_Colonne=SimTable[A_label].drop_duplicates().values.tolist()
    B_Righe=SimTable[B_label].drop_duplicates().values.tolist()
  
    SimMatrix = pd.DataFrame(np.nan,columns=A_Colonne, index=B_Righe)

    for col in A_Colonne:
        for row in B_Righe:
            s=SimTable[(SimTable[A_label]==col) &  (SimTable[B_label]==row)]['sim']
            if not(s.empty):
                SimMatrix[col][row]=s

    SimMatrix.columns.name = 'A'
    SimMatrix.index.name = 'B'
    return SimMatrix

def SimilarityMatrix(A:SchemaDoc,B:SchemaDoc):
    SimTable=SimilarityTable(A,B)
    A_Colonne=SimTable['A'].drop_duplicates().values.tolist()
    B_Righe=SimTable['B'].drop_duplicates().values.tolist()
  
    SimMatrix = pd.DataFrame(np.nan,columns=A_Colonne, index=B_Righe)

    for col in A_Colonne:
        for row in B_Righe:
            s=SimTable[(SimTable.A==col) &  (SimTable.B==row)]['sim']
            if not(s.empty):
                SimMatrix[col][row]=s

    SimMatrix.columns.name = 'A'
    SimMatrix.index.name = 'B'

    return SimMatrix

#-----------------Friend Function similarità tra SchemaDocs-------------------
def similaritySchemadocs(sda:SchemaDoc, sdb:SchemaDoc):
    DA=pd.DataFrame({'A': sda.docs_})
    DB=pd.DataFrame({'B': sdb.docs_})
    PCC = DA.assign(key=1).merge(DB.assign(key=1), on='key').drop('key', 1)
    PCC.columns=['A','B']

    ls = []
    for tup in PCC.itertuples():
        a = tup[1]
        b = tup[2]
        ls.append(getattr(sda, a).nlp.similarity(getattr(sdb, b).nlp))

    PCC['sim'] = ls

    return PCC

#------------------Matching --------------------------------------------------
def Top1(MT):
  CMT=deepcopy(MT)
  
  CMT['A_RowNo'] = CMT.sort_values(['sim'], ascending=[False]) \
             .groupby(['A']) \
             .cumcount() + 1

  CMT['B_RowNo'] = CMT.sort_values(['sim'], ascending=[False]) \
             .groupby(['B']) \
             .cumcount() + 1

  return CMT[(CMT.A_RowNo==1) & (CMT.B_RowNo==1)].drop(['A_RowNo', 'B_RowNo'], 1).sort_values(['sim'], ascending=[False])

def TopK(MT, K=2, AoB='A'):
  CMT=deepcopy(MT)
  
  CMT['RowNo'] = CMT.sort_values(['sim'], ascending=[False]) \
             .groupby([AoB]) \
             .cumcount() + 1

  return CMT[(CMT.RowNo<=K)].drop('RowNo', 1).sort_values(['A', 'sim'], ascending=[True, False])

def StableMarriage(MatchTable, threshold=0, full=False):
    MATCH = pd.DataFrame(columns=['A', 'B','sim'])
    MT=deepcopy(MatchTable)
    MT=MT.sort_values(['sim'], ascending=[False])
    while True:
        R=MT.loc[(~MT['A'].isin(MATCH['A'])) & (~MT['B'].isin(MATCH['B'])) & (MT['sim'] > threshold)]
        if len(R)==0:
            break
        x=R.iloc[0,:]
        MATCH=MATCH.append(x, ignore_index=True)

    #Aggiungo gli attributi senza corrispettivo (full join)
    if full:
        for el in set(MatchTable.A)-set(MATCH['A']):
            MATCH = MATCH.append({'A':el, 'B':'', 'sim': 0}, ignore_index=True)
        for el in set(MatchTable.B)-set(MATCH['B']):
            MATCH = MATCH.append({'A':'', 'B':el, 'sim': 0}, ignore_index=True)
    return MATCH

