from __future__ import division
from collections import defaultdict
from math import log,fabs
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

sentences = ['gato huye perro', 'perro come gato', 'ruedor huye gato','gato come raton']#['gato come','gato corre','gato juega','perro come', 'perro juega', 'perro corre']

def vocab():
	dict = defaultdict()
	dict.default_factory = lambda: len(dict)
	return dict

def get_ids(C, dict):
	for s in C:
		yield [dict[w] for w in s.split()]

voc = vocab()
C = list(get_ids(sentences,voc))



n = 2
m = len(voc)
W = (np.random.random((m,n)) - 0.5)/n
V = (np.random.random((n,m)) - 0.5)/n

def plot_words(Z,ids):
	r=0
	plt.scatter(Z[:,0],Z[:,1], marker='o', c='blue')
	for label,x,y in zip(ids, Z[:,0], Z[:,1]):
		plt.annotate(label.decode('utf8'), xy=(x,y), xytext=(-10,10), textcoords='offset points', ha= 'center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'))
		r+=1
	plt.show()
	
plot_words(W,voc)
	
def d(v_o,h):
	return fabs( np.dot(h,V.T[voc[v_o]]) ) / ( np.linalg.norm(h) * np.linalg.norm(V.T[voc[v_o]]) )

def softmax(v_o,h):
	return np.exp(-d(v_o,h)) / sum([np.exp(-d(w_j,h))  for w_j in voc])

def log2(x):
	if x != 0:
		return log(x,2)
	else:
		return 0

def loss(p,q):
	return fabs(sum([p[i]*log(p[i]/q[i]) for i in range(len(p))]))
	
def h(neigs):
	h = np.zeros(n)
	for w_k in neigs:
		h += W[voc[w_k]]
	return h/len(neigs)

def gradient(v,w,err,l_r=1.0):
	for w_i in voc:
		P_w = softmax(w_i,w)
		V.T[voc[w_i]] = V.T[voc[w_i]] - l_r * err * w

def i2h(neigs,l_r=1.0):
	for w in neigs:  
		W[voc[w]] = W[voc[w]] - l_r * (V.sum(1)/len(neigs))

loss_f = []		
perd = 10
it = 0
while perd > 1e-13:
	l_r = 1
	p_words = []
	q_words = []
	for word in voc:
		for s in sentences:
			neigs = s.split()
			if word in neigs:
				neigs.remove(word)
				p_words.append(softmax(word,h(neigs)))
				gradient(word,h(neigs),perd,l_r)
				i2h(neigs,l_r)
				q_words.append(softmax(word,h(neigs)))

	perd = loss(p_words,q_words)
	print perd
	loss_f.append(perd)
	it += 1
print it
for w in voc:
	print w,W[voc[w]]

#plot_words(W,voc)
#plt.plot(loss_f[5:])
#plt.show()
