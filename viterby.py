#-*-encoding:utf8-*-
# Metodos-analiticos-para-texto
from pickle import load
from operator import itemgetter
import numpy as np

A,B,Pi,S,O = load(open('HMM.p','r'))

text = raw_input('INP: ').split() #.encode('utf_8').split()

d = 0
tags = []
for k in range(len(text)):
	try:
		obs = B[O[text[k]]]
	except:
		obs = np.ones(len(S))*0.001
	if k == 0:
		phi,d = max( {t:Pi[i]*obs[i] for t,i in S.items()}.iteritems(), key=itemgetter(1) )
		tags.append(phi)
#		print text[k],phi,d
	else:
		ant = S[phi]
		phi,d = max( {t:d*A[ant][i]*obs[i] for t,i in S.items()}.iteritems(), key=itemgetter(1) )
		tags.append(phi)
#		print text[k],phi,d

print '[',
for j,w in enumerate(text):
	print '[{"token":"%s",' % w,
	print '"tag":"%s"},' % tags[j],
print ']'

print '%',d
