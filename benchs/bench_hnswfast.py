import faiss
import numpy as np

dim = 960
M = 64
train_number = 8000
test_number = 1000
k = 10
hnswfast_index = faiss.IndexHNSWfastFlat(dim, M)
flat_index = faiss.IndexFlat(dim)
hnswfast_index.hnsw.efSearch = 64
hnswfast_index.hnsw.efConstruction = 64
xb = np.random.random(size=(train_number, dim)).astype('float32')
xq = np.random.random(size=(test_number, dim)).astype('float32')
hnswfast_index.init_hnsw(train_number)
hnswfast_index.add(xb)
flat_index.add(xb)

D_hnswfast, I_hnswfast = hnswfast_index.search(xq, k)
D_flat, I_flat = flat_index.search(xq, k)

hnswfast_recall = 0
for i in range(test_number):
    intersect = np.intersect1d(I_hnswfast[i], I_flat[i])
    hnswfast_recall = hnswfast_recall+len(intersect)

hnswfast_recall = float(hnswfast_recall)/(k*test_number)
print('hnswfast recall {}'.format(hnswfast_recall))


import hnswlib

# Declaring index
p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

p.init_index(max_elements=train_number, ef_construction=hnswfast_index.hnsw.efConstruction, M=M)

p.set_num_threads(8)

p.add_items(xb)

p.set_ef(hnswfast_index.hnsw.efSearch)

# Query the elements for themselves and measure recall:
labels, distances = p.knn_query(xq, k=10)
hnswlib_recall = 0
for i in range(test_number):
    intersect = np.intersect1d(labels, I_flat[i])
    hnswlib_recall = hnswlib_recall+len(intersect)

hnswlib_recall = float(hnswlib_recall)/(k*test_number)
print('hnswlib recall {}'.format(hnswlib_recall))