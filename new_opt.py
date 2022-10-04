import enum
import numpy as np
import graph_tool.all as gt
from collections import namedtuple
from modules.SGD_MDS_sphere import gradient,geodesic
from modules.graph_functions import apsp

import math
import random

from modules.graph_io import write_to_json


Pair = namedtuple("Pair", "vi vj dij")

def schedule_convergent(d,t_max,eps,t_maxmax):
    w = d.copy()
    w_min,w_max = 10000, 0
    for i in range(w.shape[0]):
        for j in range(w.shape[0]):
            if i == j:
                w[i,j] = 0
            else:
                w[i,j] = d[i][j] ** -2
                w_min = min(w[i,j], w_min)
                w_max = max(w[i,j],w_max)

    eta_max = 1.0 / w_min
    eta_min = eps / w_max

    lamb = np.log(eta_max/eta_min) / (t_max-1)

    # initialize step sizes
    etas = np.zeros(t_maxmax)
    eta_switch = 1.0 / w_max
    for t in range(t_maxmax):
        eta = eta_max * np.exp(-lamb * t)
        if (eta < eta_switch): break

        etas[t] = eta

    tau = t
    for t in range(t,t_maxmax):
        eta = eta_switch / (1 + lamb*(t-tau))
        etas[t] = eta
        #etas[t] = 1e-7

    #etas = [eps for t in range(t_maxmax)]
    #print(etas)
    return etas


def sgd(Pairs,n,steps):
    X = np.random.uniform(0,1,(n,2))
    for step in steps:

        for pair in Pairs:
            i,j,dij = pair.vi, pair.vj,pair.dij
            pq = X[i]-X[j]
            mag = (pq[0]**2 + pq[1]**2) ** 0.5
            mag_grad = pq/mag
            mu = step / dij**2
            if mu > 1:
                mu = 1

            r = (mu*(mag-dij))/(2*mag)

            m = r*pq

            X[i] -= m 
            X[j] += m
    return X


def sgd_sphere(Pairs,n,steps):
    X = np.random.uniform(0,1,(n,2))
    for step in steps:

        for pair in Pairs:
            i,j,dij = pair.vi, pair.vj,pair.dij

            wc = step 
            wc = 0.5 if wc > 0.5 else wc

            delta = geodesic(X[i],X[j])
            g = gradient(X[i],X[j],1) * 2 * (1*delta-dij)
            m = wc * g

            X[i] = X[i] - m[0]
            X[j] = X[j] - m[1]

    return X    



def main():
    G = gt.load_graph_from_csv("graphs/dodecahedron_4.txt")


    #Temporary method 
    d = apsp(G)
    max_dist = d.max()
    md = np.where(d == d.max())
    K = {vi for m in md for vi in m }

    K = random.sample(tuple(K),k=min(8,len(K)))
    print(K)


    Pairs = list()
    stdist = math.pi/max_dist


    for k in K:
        dist = gt.shortest_distance(G,k)
        for i,dik in enumerate(dist):
            if i != k: Pairs.append(Pair(i,k,stdist*dik))
    for u,v in G.iter_edges():
        Pairs.append(Pair(u,v,stdist))


    ecc = [max(v) for v in d]
    radius = min(ecc)
    center = [v for v in G.iter_vertices() if ecc[v] == radius]
    center = random.sample(center,k=min(4,len(center)))

    for k in center:
        dist = gt.shortest_distance(G,k)
        for i,dik in enumerate(dist):
            if i != k: Pairs.append(Pair(i,k,stdist*dik))

    random.shuffle(Pairs)

    steps = schedule_convergent(d,30,0.01,200)
    X = sgd_sphere(Pairs,G.num_vertices(),steps)

    write_to_json(G,X)
    # pos = G.new_vp('vector<float>')
    # pos.set_2d_array(X.T)
    # gt.graph_draw(G,pos=pos)

        

    

main()