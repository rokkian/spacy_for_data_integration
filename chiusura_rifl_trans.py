""" Funzione della chiusura transitiva riflessiva
Collego tutte le coppie di attributi che hanno una relazione di correlazione

"""
import pandas as pd

def ChiusuraRiflessivaTransitiva(MatchTable):
  def transitive_closure(a):
    closure = set(a)
    while True:
        new_relations = set((x,w) for x,y in closure for q,w in closure if q == y)
        closure_until_now = closure | new_relations
        if closure_until_now == closure:
            break
        closure = closure_until_now
    return closure

  MatchTable.columns=['A','B']
  SYMMETRIC=MatchTable[['B','A']]
  SYMMETRIC.rename(columns={'B':'A','A':'B'}, inplace=True)
  SYMMETRICMatchTable=MatchTable.append(SYMMETRIC)
  
  #SYMMETRIC_TRANSITIVE_CLOSURE=transitive_closure([(str(x[1]),str(x[0])) for x in SYMMETRICMatchTable.values.tolist()])
  SYMMETRIC_TRANSITIVE_CLOSURE=transitive_closure([(x[1],x[0]) for x in SYMMETRICMatchTable.values.tolist()])
  SYMMETRIC_TRANSITIVE_CLOSURE = pd.DataFrame(data=list(SYMMETRIC_TRANSITIVE_CLOSURE), 
                        columns=['A','B'])

  return SYMMETRIC_TRANSITIVE_CLOSURE