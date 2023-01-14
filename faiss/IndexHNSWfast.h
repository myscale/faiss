#pragma once

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/HNSWfast.h>
#include <faiss/utils/utils.h>

namespace faiss {

struct IndexHNSWfast;

struct ReconstructFromNeighbors2 {
    typedef Index::idx_t idx_t;
    typedef HNSWfast::storage_idx_t storage_idx_t;

    const IndexHNSWfast& index;
    size_t M;   // number of neighbors
    size_t k;   // number of codebook entries
    size_t nsq; // number of subvectors
    size_t code_size;
    int k_reorder; // nb to reorder. -1 = all

    std::vector<float> codebook; // size nsq * k * (M + 1)

    std::vector<uint8_t> codes; // size ntotal * code_size
    size_t ntotal;
    size_t d, dsub; // derived values

    explicit ReconstructFromNeighbors2(
            const IndexHNSWfast& index,
            size_t k = 256,
            size_t nsq = 1);

    /// codes must be added in the correct order and the IndexHNSWfast
    /// must be populated and sorted
    void add_codes(size_t n, const float* x);

    size_t compute_distances(
            size_t n,
            const idx_t* shortlist,
            const float* query,
            float* distances) const;

    /// called by add_codes
    void estimate_code(const float* x, storage_idx_t i, uint8_t* code) const;

    /// called by compute_distances
    void reconstruct(storage_idx_t i, float* x, float* tmp) const;

    void reconstruct_n(storage_idx_t n0, storage_idx_t ni, float* x) const;

    /// get the M+1 -by-d table for neighbor coordinates for vector i
    void get_neighbor_table(storage_idx_t i, float* out) const;
};

/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

struct IndexHNSWfast : Index {
    typedef HNSWfast::storage_idx_t storage_idx_t;

    // the link strcuture
    HNSWfast hnsw;

    mutable bool hnsw_written = false;

    // the sequential storage
    bool own_fields;
    Index* storage;

    ReconstructFromNeighbors2* reconstruct_from_neighbors;

    explicit IndexHNSWfast(
            int d = 0,
            int M = 32,
            MetricType metric = METRIC_L2);
    explicit IndexHNSWfast(Index* storage, int M = 32);

    ~IndexHNSWfast() override;

    void add(idx_t n, const float* x) override;

    size_t remove_ids(const IDSelector& sel) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    void init_hnsw(idx_t total);

    void init_hnsw();

    void get_sorted_access_counts(std::vector<size_t>& ret, size_t& tot);
};

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

struct IndexHNSWfastFlat : IndexHNSWfast {
    IndexHNSWfastFlat();
    IndexHNSWfastFlat(int d, int M, MetricType metric = METRIC_L2);
};

/** PQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWfastPQ : IndexHNSWfast {
    IndexHNSWfastPQ();
    IndexHNSWfastPQ(
            int d,
            int pq_m,
            int bit_size = 8,
            int M = 32,
            MetricType metric = METRIC_L2);
    void train(idx_t n, const float* x) override;
};

/** SQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWfastSQ : IndexHNSWfast {
    IndexHNSWfastSQ();
    IndexHNSWfastSQ(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            int M,
            MetricType metric = METRIC_L2);
};

/** 2-level code structure with fast random access
 */
struct IndexHNSWfast2Level : IndexHNSWfast {
    IndexHNSWfast2Level();
    IndexHNSWfast2Level(Index* quantizer, size_t nlist, int m_pq, int M);

    void flip_to_ivf();

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params_in) const;
};

} // namespace faiss
