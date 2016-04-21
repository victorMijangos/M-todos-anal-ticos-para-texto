from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tsne import *
from lowDims import *
from math import isnan

def plot_words(Z,ids,mark='o',color='blue'):
	r=0
	plt.scatter(Z[:,0],Z[:,1], marker=mark, c=color)
	for label,x,y in zip(ids, Z[:,0], Z[:,1]):
		plt.annotate(label.decode('utf8'), xy=(x,y), xytext=(-1,1), textcoords='offset points', ha= 'center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.0), arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'))
		r+=1
	#plt.show()

sentences = ['gato huye perro', 'perro come gato', 'ruedor huye gato','gato come raton','ruedor come','raton come','raton huye','ruedor huye','gato come','perro come']
	
def get_vocab():
	vocab = defaultdict()
	vocab.default_factory = lambda: len(vocab)
	return vocab
	
def get_BoW(S,vocab):
	for s in S:
		yield [vocab[w] for w in s.split()]
		
vocab = get_vocab()
bow = list(get_BoW(sentences,vocab))

n = len(vocab)
m = 100

W = (np.random.rand(n,m)- 0.5) / n
V = (np.random.random((m,n)) - 0.5) / n
#plot_words(svd(W,2),vocab.keys(),'v','yellow')

def d(x,y):
	return np.dot(x,y)

def softmax(x,h):
	return np.exp(-d(x,h))/np.exp( sum([-d(V.T[i],h) for i in range(n)]) )
	
def h(neigs):
	h = np.zeros(m)
	for w_k in neigs:
		h += W[vocab[w_k]]
	return h/len(neigs)
	
j = 0
l = 0.01
err = 1
while err != 0.0:
	for obj_w in vocab:
	
		for s in sentences:
			neigs = s.split()
			
			for word in vocab:
				if word in neigs:
					neigs.remove(word)
					H = h(neigs)
					p = softmax(V.T[vocab[word]],H)
					t = 1 if word == obj_w else 0
					err = t - p
					V.T[vocab[word]] -= l*err*H
					neigs.append(word)
					
			for word in neigs:
				W[vocab[word]] -= (1./len(neigs))*l*V.sum(1)
				
	print err
	if isnan(err):
		break
	j += 1
	
plot_words(svd(W,2),vocab.keys())

plt.show()
