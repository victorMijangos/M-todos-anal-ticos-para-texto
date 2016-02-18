#-*-encoding:utf8-*-
# Metodos-analiticos-para-texto
from pickle import load
from operator import itemgetter
import numpy as np

#Importa el modelo de entrenamiento
A,B,Pi,S,O = load(open('HMM.p','r'))

#Texto a analizar
text = raw_input('INP: ').split() #.encode('utf_8').split()

#Algoritmo de Viterbi
d = 0
tags = []
for k in range(len(text)):
	try:
		#Asigna el vector de probabilidades a una observacion
		obs = B[O[text[k]]]
	except:
		#Si no encuentra alguna palabra en los datos de entrenamiento
		obs = np.ones(len(S))
	if k == 0:
	#Inicialización
		phi,d = max( {t:Pi[i]*obs[i] for t,i in S.items()}.iteritems(), key=itemgetter(1) )
		tags.append(phi)
	else:
	#Induccion
		ant = S[phi]
		phi,d = max( {t:d*A[ant][i]*obs[i] for t,i in S.items()}.iteritems(), key=itemgetter(1) )
		tags.append(phi)

#Imprime en formato JSON
print '[',
for j,w in enumerate(text):
	print '[{"token":"%s",' % w,
	print '"tag":"%s"},' % tags[j],
print ']'

print '%',d
