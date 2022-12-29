import faiss
import numpy as np
np.random.seed(100)
dim = 960
M = 32
train_number = 1000
test_number = 100
k = 10
hnswfast_index = faiss.IndexHNSWfastFlat(dim, M)
hnsw_index = faiss.IndexHNSWFlat(dim, M)
flat_index = faiss.IndexFlat(dim)
faiss.omp_set_num_threads(1)
xb = np.random.random(size=(train_number, dim)).astype('float32')
xq = np.random.random(size=(test_number, dim)).astype('float32')

# hnswfast
hnswfast_index.hnsw.efConstruction = 32
hnswfast_index.init_hnsw(train_number)
hnswfast_index.add(xb)
flat_index.add(xb)
hnswfast_index.hnsw.efSearch = 32

D_hnswfast, I_hnswfast = hnswfast_index.search(xq, k)
D_flat, I_flat = flat_index.search(xq, k)

hnswfast_recall = 0
for i in range(test_number):
    intersect = np.intersect1d(I_hnswfast[i], I_flat[i])
    hnswfast_recall = hnswfast_recall+len(intersect)

hnswfast_recall = float(hnswfast_recall)/(k*test_number)
print('hnswfast recall {}'.format(hnswfast_recall))

# hnsw faiss
hnsw_index.hnsw.efConstruction = hnswfast_index.hnsw.efConstruction
hnsw_index.hnsw.efSearch = hnswfast_index.hnsw.efSearch
hnsw_index.add(xb)
D_hnsw, I_hnsw = hnsw_index.search(xq, k)
hnsw_recall = 0
for i in range(test_number):
    intersect = np.intersect1d(I_hnsw[i], I_flat[i])
    hnsw_recall = hnsw_recall+len(intersect)

hnsw_recall = float(hnsw_recall)/(k*test_number)
print('hnsw recall {}'.format(hnsw_recall))

# hnswlib
import hnswlib
p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
p.init_index(max_elements=train_number, ef_construction=hnswfast_index.hnsw.efConstruction, M=M)
p.set_num_threads(8)
p.add_items(xb)
p.set_ef(hnswfast_index.hnsw.efSearch)

labels, distances = p.knn_query(xq, k=k)
hnswlib_recall = 0
for i in range(test_number):
    intersect = np.intersect1d(labels[i], I_flat[i])
    hnswlib_recall = hnswlib_recall+len(intersect)

hnswlib_recall = float(hnswlib_recall)/(k*test_number)
print('hnswlib recall {}'.format(hnswlib_recall))

# hnswfast sq
hnswfast_sq_index = faiss.IndexHNSWfastSQ(dim, faiss.ScalarQuantizer.QT_8bit, M)
hnswfast_sq_index.hnsw.efConstruction = hnswfast_index.hnsw.efConstruction
hnswfast_sq_index.init_hnsw(train_number)
hnswfast_sq_index.train(xb)
hnswfast_sq_index.add(xb)

hnswfast_sq_index.hnsw.efSearch = hnswfast_index.hnsw.efSearch

D_hnswfast_sq, I_hnswfast_sq = hnswfast_sq_index.search(xq, k)

hnswfast_sq_recall = 0
for i in range(test_number):
    intersect = np.intersect1d(I_hnswfast_sq[i], I_flat[i])
    hnswfast_sq_recall = hnswfast_sq_recall+len(intersect)

hnswfast_sq_recall = float(hnswfast_sq_recall)/(k*test_number)
print('hnswfast sq recall {}'.format(hnswfast_sq_recall))

# hnswfast pq
hnswfast_pq_index = faiss.IndexHNSWfastPQ(dim, 20, M)
hnswfast_pq_index.hnsw.efConstruction = hnswfast_index.hnsw.efConstruction
hnswfast_pq_index.init_hnsw(train_number)
hnswfast_pq_index.train(xb)
hnswfast_pq_index.add(xb)

hnswfast_pq_index.hnsw.efSearch = hnswfast_index.hnsw.efSearch

D_hnswfast_pq, I_hnswfast_pq = hnswfast_pq_index.search(xq, k)

hnswfast_pq_recall = 0
for i in range(test_number):
    intersect = np.intersect1d(I_hnswfast_pq[i], I_flat[i])
    hnswfast_pq_recall = hnswfast_pq_recall+len(intersect)

hnswfast_pq_recall = float(hnswfast_pq_recall)/(k*test_number)
print('hnswfast pq recall {}'.format(hnswfast_pq_recall))