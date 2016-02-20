# M-todos-anal-ticos-para-texto
from __future__ import division
from itertools import combinations
import numpy as np
from cmath import *
from math import isnan

#Proyeccion ortogonal de v2 sobre v1
def proj(v1, v2):
	return (np.dot(v2, v1)/np.dot(v1,v1))*v1
 
def gs(V):
  #Proceso de ortogonalizacion
	m,n = V.shape
	E = np.zeros((m,n))
	E[0] = V[0]
	for k in range(m-1):
		proj_k = proj(E[0],V[k+1])
		if k > 0:
			proj_k += proj(E[k],V[k+1])

		E[k+1] = V[k+1] - proj_k
	
	#Normalizacion de los vectores	
	for i,e in enumerate(E):
		E[i] = e/np.linalg.norm(e)

	return E
