import faiss
import numpy as np
import time

np.random.seed(100)
dim = 960
M = 64
train_number = 8000
test_number = 1000
k = 10
hnswfast_index = faiss.IndexHNSWfastFlat(dim, M)
hnsw_index = faiss.IndexHNSWFlat(dim, M)
flat_index = faiss.IndexFlat(dim)

xb = np.random.random(size=(train_number, dim)).astype('float32')
xq = np.random.random(size=(test_number, dim)).astype('float32')

rs = np.random.RandomState(123)
subset = rs.choice([i for i in range(len(xb))], 900, replace=False).astype("int64")
print('id select(subset) size {}'.format(len(subset)))

hnswfast_index.hnsw.efConstruction = 64
hnswfast_index.hnsw.efSearch = 64 * 2
flat_index.add(xb)
start = time.time()
D_flat, I_flat = flat_index.search(xq, k)
end = time.time()
print('flat: {}'.format(end - start))

# hnsw faiss
#def test_hnsw_faiss():
hnsw_index.hnsw.efConstruction = hnswfast_index.hnsw.efConstruction
hnsw_index.hnsw.efSearch = hnswfast_index.hnsw.efSearch
hnsw_index.add(xb)
hnswfast_index.init_hnsw(train_number)
hnswfast_index.add(xb)

def test_hnswlib():
# hnswlib
    import hnswlib
    p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
    p.init_index(max_elements=train_number, ef_construction=hnswfast_index.hnsw.efConstruction, M=M)
    p.add_items(xb)
    p.set_ef(hnswfast_index.hnsw.efSearch)

    p.set_num_threads(1)
    start = time.time()
    labels, distances = p.knn_query(xq, k=k)
    end = time.time()
    print('hnswlib time: {}'.format(end - start))

    hnswlib_recall = 0
    for i in range(test_number):
        intersect = np.intersect1d(labels[i], I_flat[i])
        hnswlib_recall = hnswlib_recall+len(intersect)

    hnswlib_recall = float(hnswlib_recall)/(k*test_number)
    print('hnswlib recall {}'.format(hnswlib_recall))

test_hnswlib()

def test_faiss_hnsw():
    faiss.omp_set_num_threads(1)
    start = time.time()
    D_hnsw, I_hnsw = hnsw_index.search(xq, k)
    end = time.time()
    print('faiss hnsw time: {}'.format(end - start))

    hnsw_recall = 0
    for i in range(test_number):
        intersect = np.intersect1d(I_hnsw[i], I_flat[i])
        hnsw_recall = hnsw_recall+len(intersect)

    hnsw_recall = float(hnsw_recall)/(k*test_number)
    print('hnsw recall {}'.format(hnsw_recall))
test_faiss_hnsw()
# hnswfast
def test_hnswfast(hnswfast_index_):
    faiss.omp_set_num_threads(1)
    start = time.time()
    D_hnswfast, I_hnswfast = hnswfast_index_.search(xq, k)
    end = time.time()
    print('hnsw fast time: {}'.format(end - start))

    hnswfast_recall = 0
    for i in range(test_number):
        intersect = np.intersect1d(I_hnswfast[i], I_flat[i])
        hnswfast_recall = hnswfast_recall+len(intersect)

    hnswfast_recall = float(hnswfast_recall)/(k*test_number)
    print('hnswfast recall {}'.format(hnswfast_recall))

test_hnswfast(hnswfast_index)
faiss.write_index(hnswfast_index, '/tmp/hnswfast')
load_hnswfast_index = faiss.read_index('/tmp/hnswfast')
test_hnswfast(load_hnswfast_index)
exit(0)

# hnswfast sq
hnswfast_sq_index = faiss.IndexHNSWfastSQ(dim, faiss.ScalarQuantizer.QT_8bit, M)
hnswfast_sq_index.hnsw.efConstruction = hnswfast_index.hnsw.efConstruction
hnswfast_sq_index.init_hnsw(train_number)
hnswfast_sq_index.train(xb)
hnswfast_sq_index.add(xb)

def test_hnswfast_sq():
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

# create id selector
bitmap = np.zeros(len(xb), dtype=bool)
bitmap[subset] = True
#print('bitmap {}'.format(bitmap))
bitmap = np.packbits(bitmap, bitorder='little')
sel = faiss.IDSelectorBitmap(bitmap)
# flat filtered
params = faiss.SearchParameters()
params.sel = sel
D_flat_filtered, I_flat_filterd = flat_index.search(xq, k, params=params)

def hnswfast_sq_filter():
    # hnswfast sq filter
    params = faiss.SearchParametersHNSW()
    params.sel = sel
    params.efSearch = hnswfast_index.hnsw.efSearch
    D_hnswfast_sq_filter, I_hnswfast_sq_filter = hnswfast_sq_index.search(xq, k, params=params)

    hnswfast_sq_filter_recall = 0
    for i in range(test_number):
        intersect = np.intersect1d(I_hnswfast_sq_filter[i], I_flat_filterd[i])
        hnswfast_sq_filter_recall = hnswfast_sq_filter_recall+len(intersect)

    hnswfast_sq_filter_recall = float(hnswfast_sq_filter_recall)/(k*test_number)
    print('hnswfast sq with filter recall {}'.format(hnswfast_sq_filter_recall))
    #print('{}'.format(I_hnswfast_sq_filter))
    #print('{}'.format(I_hnswfast_sq))

    # hnsw sq filter
    params = faiss.SearchParametersHNSW()
    params.sel = sel
    params.efSearch = hnswfast_index.hnsw.efSearch
    D_hnsw, I_hnsw_filterd = hnsw_index.search(xq, k, params=params)

    hnsw_recall = 0
    for i in range(test_number):
        intersect = np.intersect1d(I_hnsw_filterd[i], I_flat_filterd[i])
        hnsw_recall = hnsw_recall+len(intersect)

    hnsw_recall = float(hnsw_recall)/(k*test_number)
    print('hnsw faiss with filter recall {}'.format(hnsw_recall))
test_hnswfast_sq()
hnswfast_sq_filter()