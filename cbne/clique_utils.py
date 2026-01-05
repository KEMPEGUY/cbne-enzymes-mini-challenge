# cbne/clique_utils.py
import numpy as np
import networkx as nx

def repl(face, i, newv):
    L = list(face)
    L[i] = newv
    L.sort()
    return tuple(L)

def _conn_except(G, face, v, i):
    for j, u in enumerate(face):
        if j != i and not G.has_edge(v, u):
            return False
    return True

def neigh_sign(face, i, v):
    before = face[:i]
    after = face[i+1:]
    count = 0
    for u in before:
        if u > v: count += 1
    for u in after:
        if u < v: count += 1
    return -1 if count % 2 else 1

def faithful_neighbours(G, face):
    out = []
    for v in range(G.n):
        if v in face: continue
        for i in range(len(face)):
            if _conn_except(G, face, v, i):
                out.append((repl(face, i, v), neigh_sign(face, i, v)))
    return out

def enumerate_k_faces(G, k):
    return [tuple(sorted(c)) for c in nx.find_cliques(G) if len(c) == k + 1]

def number_of_up_neighbours(G, sigma):
    return sum(1 for v in range(G.n) if v not in sigma and all(G.has_edge(v, u) for u in sigma))

def build_clique_structures(G, k):
    G.n = G.number_of_nodes()
    faces = enumerate_k_faces(G, k)
    F = len(faces)
    face_to_idx = {f: i for i, f in enumerate(faces)}

    diag = np.zeros(F, dtype=np.float32)
    off = [0]
    idx = []
    sgn = []

    for j, sigma in enumerate(faces):
        up = number_of_up_neighbours(G, sigma)
        diag[j] = 1.0 - float(up + len(sigma)) / float(G.n)

        neigh = faithful_neighbours(G, sigma)
        for tau, sg in neigh:
            idx.append(face_to_idx.get(tau, 0))
            sgn.append(sg)
        off.append(len(idx))

    return (faces,
            np.array(diag, np.float32),
            np.array(off, np.int64),
            np.array(idx, np.int64),
            np.array(sgn, np.int8),
            np.array(off[1:], np.int64) - np.array(off[:-1], np.int64))

