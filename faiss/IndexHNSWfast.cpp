#include <faiss/IndexHNSWfast.h>

#include <omp.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <queue>
#include <unordered_set>

#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef __SSE__
#endif

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <iostream>

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

using idx_t = Index::idx_t;
using MinimaxHeap = HNSWfast::MinimaxHeap;
using storage_idx_t = HNSWfast::storage_idx_t;
using NodeDistFarther = HNSWfast::NodeDistFarther;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
            : basedis(basedis) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    virtual ~NegativeDistanceComputer() {
        delete basedis;
    }
};

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (storage->metric_type == METRIC_INNER_PRODUCT) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

void hnsw_add_vertices(
        IndexHNSWfast& index_hnsw,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_hnsw.d;
    HNSWfast& hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hnsw_add_vertices: adding %ld elements on top of %ld "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = hnsw.max_level;

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    { // perform add
        auto tas = getmillisecs();
        DistanceComputer* dis0 = storage_distance_computer(index_hnsw.storage);
        ScopeDeleter1<DistanceComputer> del0(dis0);

        dis0->set_query(x);
        hnsw.addPoint(*dis0, hnsw.levels[n0], n0);

#pragma omp parallel for
        for (int i = 1; i < n; ++i) {
            DistanceComputer* dis =
                    storage_distance_computer(index_hnsw.storage);
            ScopeDeleter1<DistanceComputer> del(dis);
            dis->set_query(x + i * d);
            hnsw.addPoint(*dis, hnsw.levels[n0 + i], i + n0);
        }
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }
}

} // namespace

/**************************************************************
 * IndexHNSWfast implementation
 **************************************************************/

IndexHNSWfast::IndexHNSWfast(int d, int M, MetricType metric)
        : Index(d, metric),
          hnsw(M),
          own_fields(false),
          storage(nullptr),
          reconstruct_from_neighbors(nullptr) {}

IndexHNSWfast::IndexHNSWfast(Index* storage, int M)
        : Index(storage->d, storage->metric_type),
          hnsw(M),
          own_fields(false),
          storage(storage),
          reconstruct_from_neighbors(nullptr) {}

IndexHNSWfast::~IndexHNSWfast() {
    if (own_fields) {
        delete storage;
    }
}

void IndexHNSWfast::init_hnsw(idx_t total) {
    hnsw.init(total);
}

void IndexHNSWfast::init_hnsw() {
    hnsw.init(ntotal);
}

void IndexHNSWfast::get_sorted_access_counts(
        std::vector<size_t>& ret,
        size_t& tot) {
    return;
}

void IndexHNSWfast::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWfastFlat (or variants) instead of IndexHNSWfast directly");
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexHNSWfast::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const

{
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWfastFlat (or variants) instead of IndexHNSWfast directly");
    size_t nreorder = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * d * hnsw.efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

/// Disable omp for more consistent results
//#pragma omp parallel reduction(+ : nreorder)
        {
            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

//#pragma omp for
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                hnsw.searchKnn(
                        *dis,
                        k,
                        idxi,
                        simi,
                        dynamic_cast<const SearchParametersHNSW*>(params_in));

                if (reconstruct_from_neighbors &&
                    reconstruct_from_neighbors->k_reorder != 0) {
                    int k_reorder = reconstruct_from_neighbors->k_reorder;
                    if (k_reorder == -1 || k_reorder > k)
                        k_reorder = k;

                    nreorder += reconstruct_from_neighbors->compute_distances(
                            k_reorder, idxi, x + i * d, simi);

                    // sort top k_reorder
                    maxheap_heapify(
                            k_reorder, simi, idxi, simi, idxi, k_reorder);
                    maxheap_reorder(k_reorder, simi, idxi);
                }
            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSWfast::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWfastFlat (or variants) instead of IndexHNSWfast directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;

    storage->add(n, x);
    ntotal = storage->ntotal;

    hnsw_add_vertices(*this, n0, n, x, verbose, hnsw.levels.size() == ntotal);
}

void IndexHNSWfast::reset() {
    hnsw.reset();
    storage->reset();
    ntotal = 0;
}

void IndexHNSWfast::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

size_t IndexHNSWfast::remove_ids(const IDSelector& sel) {
    size_t nremove = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member(i)) {
            nremove++;
        }
    }
    if (nremove)
        hnsw.has_deletion = true;
    return nremove;
}

/**************************************************************
 * ReconstructFromNeighbors implementation
 **************************************************************/

ReconstructFromNeighbors2::ReconstructFromNeighbors2(
        const IndexHNSWfast& index,
        size_t k,
        size_t nsq)
        : index(index), k(k), nsq(nsq) {
    M = index.hnsw.M << 1;
    FAISS_ASSERT(k <= 256);
    code_size = k == 1 ? 0 : nsq;
    ntotal = 0;
    d = index.d;
    FAISS_ASSERT(d % nsq == 0);
    dsub = d / nsq;
    k_reorder = -1;
}

void ReconstructFromNeighbors2::reconstruct(
        storage_idx_t i,
        float* x,
        float* tmp) const {
    const HNSWfast& hnsw = index.hnsw;
    int* cur_links = hnsw.get_neighbor_link(i, 0);
    int* cur_neighbors = cur_links + 1;
    auto cur_neighbor_num = hnsw.get_neighbors_num(cur_links);

    if (k == 1 || nsq == 1) {
        const float* beta;
        if (k == 1) {
            beta = codebook.data();
        } else {
            int idx = codes[i];
            beta = codebook.data() + idx * (M + 1);
        }

        float w0 = beta[0]; // weight of image itself
        index.storage->reconstruct(i, tmp);

        for (int l = 0; l < d; l++)
            x[l] = w0 * tmp[l];

        for (auto j = 0; j < cur_neighbor_num; ++j) {
            storage_idx_t ji = cur_neighbors[j];
            if (ji < 0)
                ji = i;
            float w = beta[j + 1];
            index.storage->reconstruct(ji, tmp);
            for (int l = 0; l < d; l++)
                x[l] += w * tmp[l];
        }
    } else if (nsq == 2) {
        int idx0 = codes[2 * i];
        int idx1 = codes[2 * i + 1];

        const float* beta0 = codebook.data() + idx0 * (M + 1);
        const float* beta1 = codebook.data() + (idx1 + k) * (M + 1);

        index.storage->reconstruct(i, tmp);

        float w0;

        w0 = beta0[0];
        for (int l = 0; l < dsub; l++)
            x[l] = w0 * tmp[l];

        w0 = beta1[0];
        for (int l = dsub; l < d; l++)
            x[l] = w0 * tmp[l];

        for (auto j = 0; j < cur_neighbor_num; ++j) {
            storage_idx_t ji = cur_neighbors[j];
            if (ji < 0)
                ji = i;
            index.storage->reconstruct(ji, tmp);
            float w;
            w = beta0[j + 1];
            for (int l = 0; l < dsub; l++)
                x[l] += w * tmp[l];

            w = beta1[j + 1];
            for (int l = dsub; l < d; l++)
                x[l] += w * tmp[l];
        }
    } else {
        const float* betas[nsq];
        {
            const float* b = codebook.data();
            const uint8_t* c = &codes[i * code_size];
            for (int sq = 0; sq < nsq; sq++) {
                betas[sq] = b + (*c++) * (M + 1);
                b += (M + 1) * k;
            }
        }

        index.storage->reconstruct(i, tmp);
        {
            int d0 = 0;
            for (int sq = 0; sq < nsq; sq++) {
                float w = *(betas[sq]++);
                int d1 = d0 + dsub;
                for (int l = d0; l < d1; l++) {
                    x[l] = w * tmp[l];
                }
                d0 = d1;
            }
        }

        for (auto j = 0; j < cur_neighbor_num; ++j) {
            storage_idx_t ji = cur_neighbors[j];
            if (ji < 0)
                ji = i;

            index.storage->reconstruct(ji, tmp);
            int d0 = 0;
            for (int sq = 0; sq < nsq; sq++) {
                float w = *(betas[sq]++);
                int d1 = d0 + dsub;
                for (int l = d0; l < d1; l++) {
                    x[l] += w * tmp[l];
                }
                d0 = d1;
            }
        }
    }
}

void ReconstructFromNeighbors2::reconstruct_n(
        storage_idx_t n0,
        storage_idx_t ni,
        float* x) const {
#pragma omp parallel
    {
        std::vector<float> tmp(index.d);
#pragma omp for
        for (storage_idx_t i = 0; i < ni; i++) {
            reconstruct(n0 + i, x + i * index.d, tmp.data());
        }
    }
}

size_t ReconstructFromNeighbors2::compute_distances(
        size_t n,
        const idx_t* shortlist,
        const float* query,
        float* distances) const {
    std::vector<float> tmp(2 * index.d);
    size_t ncomp = 0;
    for (int i = 0; i < n; i++) {
        if (shortlist[i] < 0)
            break;
        reconstruct(shortlist[i], tmp.data(), tmp.data() + index.d);
        distances[i] = fvec_L2sqr(query, tmp.data(), index.d);
        ncomp++;
    }
    return ncomp;
}

void ReconstructFromNeighbors2::get_neighbor_table(storage_idx_t i, float* tmp1)
        const {
    const HNSWfast& hnsw = index.hnsw;
    int* cur_links = hnsw.get_neighbor_link(i, 0);
    int* cur_neighbors = cur_links + 1;
    auto cur_neighbor_num = hnsw.get_neighbors_num(cur_links);
    size_t d = index.d;

    index.storage->reconstruct(i, tmp1);

    for (auto j = 0; j < cur_neighbor_num; ++j) {
        storage_idx_t ji = cur_neighbors[j];
        if (ji < 0)
            ji = i;
        index.storage->reconstruct(ji, tmp1 + (j + 1) * d);
    }
}

/// called by add_codes
void ReconstructFromNeighbors2::estimate_code(
        const float* x,
        storage_idx_t i,
        uint8_t* code) const {
    // fill in tmp table with the neighbor values
    float* tmp1 = new float[d * (M + 1) + (d * k)];
    float* tmp2 = tmp1 + d * (M + 1);
    ScopeDeleter<float> del(tmp1);

    // collect coordinates of base
    get_neighbor_table(i, tmp1);

    for (size_t sq = 0; sq < nsq; sq++) {
        int d0 = sq * dsub;

        {
            FINTEGER ki = k, di = d, m1 = M + 1;
            FINTEGER dsubi = dsub;
            float zero = 0, one = 1;

            sgemm_("N",
                   "N",
                   &dsubi,
                   &ki,
                   &m1,
                   &one,
                   tmp1 + d0,
                   &di,
                   codebook.data() + sq * (m1 * k),
                   &m1,
                   &zero,
                   tmp2,
                   &dsubi);
        }

        float min = HUGE_VAL;
        int argmin = -1;
        for (size_t j = 0; j < k; j++) {
            float dis = fvec_L2sqr(x + d0, tmp2 + j * dsub, dsub);
            if (dis < min) {
                min = dis;
                argmin = j;
            }
        }
        code[sq] = argmin;
    }
}

void ReconstructFromNeighbors2::add_codes(size_t n, const float* x) {
    if (k == 1) { // nothing to encode
        ntotal += n;
        return;
    }
    codes.resize(codes.size() + code_size * n);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        estimate_code(
                x + i * index.d,
                ntotal + i,
                codes.data() + (ntotal + i) * code_size);
    }
    ntotal += n;
    FAISS_ASSERT(codes.size() == ntotal * code_size);
}

/**************************************************************
 * IndexHNSWfastFlat implementation
 **************************************************************/

IndexHNSWfastFlat::IndexHNSWfastFlat() {
    is_trained = true;
}

IndexHNSWfastFlat::IndexHNSWfastFlat(int d, int M, MetricType metric)
        : IndexHNSWfast(new IndexFlat(d, 0, metric), M) {
    own_fields = true;
    is_trained = true;
}

/**************************************************************
 * IndexHNSWfastPQ implementation
 **************************************************************/

IndexHNSWfastPQ::IndexHNSWfastPQ() {}

IndexHNSWfastPQ::IndexHNSWfastPQ(
        int d,
        int pq_m,
        int bit_size,
        int M,
        MetricType metric)
        : IndexHNSWfast(new IndexPQ(d, pq_m, bit_size, metric), M) {
    own_fields = true;
    is_trained = false;
}

void IndexHNSWfastPQ::train(idx_t n, const float* x) {
    IndexHNSWfast::train(n, x);
    (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
}

/**************************************************************
 * IndexHNSWfastSQ implementation
 **************************************************************/

IndexHNSWfastSQ::IndexHNSWfastSQ(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M,
        MetricType metric)
        : IndexHNSWfast(new IndexScalarQuantizer(d, qtype, metric), M) {
    own_fields = true;
    is_trained = false;
}

IndexHNSWfastSQ::IndexHNSWfastSQ() {}

/**************************************************************
 * IndexHNSWfast2Level implementation
 **************************************************************/

IndexHNSWfast2Level::IndexHNSWfast2Level(
        Index* quantizer,
        size_t nlist,
        int m_pq,
        int M)
        : IndexHNSWfast(new Index2Layer(quantizer, nlist, m_pq), M) {
    own_fields = true;
    is_trained = false;
}

IndexHNSWfast2Level::IndexHNSWfast2Level() = default;

namespace {

// same as search_from_candidates but uses v
// visno -> is in result list
// visno + 1 -> in result list + in candidates
int search_from_candidates_2(
        const HNSWfast& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedList& vt,
        int level,
        int nres_in = 0) {
    int nres = nres_in;
    int ndis = 0;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        FAISS_ASSERT(v1 >= 0);
        vt.mass[v1] = vt.curV + 1;
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        int* cur_links = hnsw.get_neighbor_link(v0, level);
        int* cur_neighbors = cur_links + 1;
        auto cur_neighbor_num = hnsw.get_neighbors_num(cur_links);

        for (auto j = 0; j < cur_neighbor_num; ++j) {
            int v1 = cur_neighbors[j];
            if (v1 < 0)
                break;
            if (vt.mass[v1] == vt.curV + 1) {
                // nothing to do
            } else {
                ndis++;
                float d = qdis(v1);
                candidates.push(v1, d);

                // never seen before --> add to heap
                if (vt.mass[v1] < vt.curV) {
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_pop(nres--, D, I);
                        faiss::maxheap_push(++nres, D, I, d, v1);
                    }
                }
                vt.mass[v1] = vt.curV + 1;
            }
        }

        nstep++;
        if (nstep > hnsw.efSearch) {
            break;
        }
    }
    return nres;
}

} // namespace

void IndexHNSWfast2Level::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* param) const {
    if (dynamic_cast<const Index2Layer*>(storage)) {
        IndexHNSWfast::search(n, x, k, distances, labels, param);

    } else { // "mixed" search

        const IndexIVFPQ* index_ivfpq =
                dynamic_cast<const IndexIVFPQ*>(storage);

        int nprobe = index_ivfpq->nprobe;

        std::unique_ptr<idx_t[]> coarse_assign(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        index_ivfpq->quantizer->search(
                n, x, nprobe, coarse_dis.get(), coarse_assign.get());

        index_ivfpq->search_preassigned(
                n,
                x,
                k,
                coarse_assign.get(),
                coarse_dis.get(),
                distances,
                labels,
                false);

#pragma omp parallel
        {
            VisitedList vt(ntotal);
            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

            const int candidates_size = 1;
            MinimaxHeap candidates(candidates_size);

#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                // mark all inverted list elements as visited

                for (int j = 0; j < nprobe; j++) {
                    idx_t key = coarse_assign[j + i * nprobe];
                    if (key < 0)
                        break;
                    size_t list_length = index_ivfpq->get_list_size(key);
                    const idx_t* ids = index_ivfpq->invlists->get_ids(key);

                    for (int jj = 0; jj < list_length; jj++) {
                        vt.set(ids[jj]);
                    }
                }

                candidates.clear();
                // copy the upper_beam elements to candidates list

                int search_policy = 2;

                if (search_policy == 1) {
                    for (int j = 0; j < candidates_size && j < k; j++) {
                        if (idxi[j] < 0)
                            break;
                        candidates.push(idxi[j], simi[j]);
                        // search_from_candidates adds them back
                        idxi[j] = -1;
                        simi[j] = HUGE_VAL;
                    }

                    // reorder from sorted to heap
                    maxheap_heapify(k, simi, idxi, simi, idxi, k);

                    // removed from HNSWfast, but still available in HNSW
                    //                    hnsw.search_from_candidates(
                    //                      *dis, k, idxi, simi,
                    //                      candidates, vt, 0, k
                    //                    );

                    vt.advance();

                } else if (search_policy == 2) {
                    for (int j = 0; j < candidates_size && j < k; j++) {
                        if (idxi[j] < 0)
                            break;
                        candidates.push(idxi[j], simi[j]);
                    }

                    // reorder from sorted to heap
                    maxheap_heapify(k, simi, idxi, simi, idxi, k);

                    search_from_candidates_2(
                            hnsw, *dis, k, idxi, simi, candidates, vt, 0, k);
                    vt.advance();
                    vt.advance();
                }

                maxheap_reorder(k, simi, idxi);
            }
        }
    }
}

void IndexHNSWfast2Level::flip_to_ivf() {
    Index2Layer* storage2l = dynamic_cast<Index2Layer*>(storage);

    FAISS_THROW_IF_NOT(storage2l);

    IndexIVFPQ* index_ivfpq = new IndexIVFPQ(
            storage2l->q1.quantizer,
            d,
            storage2l->q1.nlist,
            storage2l->pq.M,
            8);
    index_ivfpq->pq = storage2l->pq;
    index_ivfpq->is_trained = storage2l->is_trained;
    index_ivfpq->precompute_table();
    index_ivfpq->own_fields = storage2l->q1.own_fields;
    storage2l->transfer_to_IVFPQ(*index_ivfpq);
    index_ivfpq->make_direct_map(true);

    storage = index_ivfpq;
    delete storage2l;
}

} // namespace faiss
