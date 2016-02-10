#-*-encoding:utf8-*-
from __future__ import division
from json import load
from sys import argv
from collections import Counter,defaultdict
import numpy as np
from pickle import dump

#Abre el archivo tipo JSON
file = load(open(argv[1],'r'))["corpus"]
docs = [d["document"] for d in file]

#Genera un diccionario que despues se llena con datos del corpus
def voc():
	dict = defaultdict()
	dict.default_factory = lambda: len(dict)
	return dict

#Llena el diccionario con datos del corpus	
def get_ids(C,dict):
	yield [dict[w] for w in C]

#Se hace una lista con las secuencias
chain = []
obs = []
#La funcion counter aumenta 1 cada vez que encuentra el mismo elemento
S_f = Counter()
O_f = Counter()
P_f = Counter()
for e in docs:
	P_f[ e[0]["tag"] ] += 1
	for word in e:
		tag = word["tag"]
		token = word["token"]
		S_f[tag] += 1 
		O_f[token] += 1
		chain.append(tag)
		obs.append((token,tag))

#Dimensiones de las matrices
n = len(S_f)
m = len(O_f)

#Smoothing de Lindstone
def pr (x, cond, N, l=1):
	return (x+l) / (cond + l*N)	

#Se generan matrices llenas de 0
A = np.zeros((n,n))
B = np.zeros((m,n))
Pi = np.zeros(n)

#S_v son los estados o etiquetas y se genera un id
S_v = voc()
list(get_ids(chain,S_v))[0]

#Se llena el vector de probabilidades iniciales
for t in S_v.keys():
	Pi[S_v[t]] = pr(P_f[t],len(docs),n)

#Se llena la matriz de transicion
chains = Counter(zip(chain,chain[1:]))
for (t,t_ant), c_ws in chains.iteritems():
	N = Counter([i[0] for i in chains])
	A[S_v[t],S_v[t_ant]] =  pr(c_ws,S_f[t],N[t])


#Se genera una lista de coincidencias de tags y observaciones
c_so = Counter(obs)

#O_v son las observaciones con un id asociado
O_v = voc()
list(get_ids(O_f.keys(),O_v))[0]

#Se genera la matriz de observaciones
for o in O_f.keys():
	for s in S_v.keys():
		B[O_v[o], S_v[s]] = pr(c_so[(o,s)],O_f[o],n)

#Se guarda el modelo aprendido
f = open('HMM.p','w')
HMM = [A,B,Pi,dict(S_v),dict(O_v)]
dump(HMM,f)
