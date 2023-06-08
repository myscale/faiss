#pragma once

#include <algorithm>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <omp.h>
//#include "/usr/local/Cellar/libomp/13.0.0/include/omp.h"

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include <SearchIndex/VectorIndex.h>

namespace faiss {

/** Implementation of the Hierarchical Navigable Small World
 * datastructure.
 *
 * Efficient and robust approximate nearest neighbor search using
 * Hierarchical Navigable Small World graphs
 *
 *  Yu. A. Malkov, D. A. Yashunin, arXiv 2017
 *
 * This implmentation is heavily influenced by the hnswlib
 * implementation by Yury Malkov and Leonid Boystov
 * (https://github.com/searchivarius/nmslib/hnswlib)
 *
 * The HNSW object stores only the neighbor link structure, see
 * IndexHNSW.h for the full index object.
 */

struct DistanceComputer; // from AuxIndexStructures
class VisitedListPool;

class HNSWfast {
   public:
    /// internal storage of vectors (32 bits: this is expensive)
    typedef int storage_idx_t;

    mutable bool hnsw_header_written = false;
    mutable bool level_size_written = false;
    mutable bool level0_links_written = false;
    mutable bool bitmap_written = false;
    mutable size_t current_level0_links_writting = 0;
    mutable size_t current_link_list_writting = 0;

    /// Faiss results are 64-bit
    typedef Index::idx_t idx_t;

    typedef std::pair<float, storage_idx_t> Node;

    /** Heap structure that allows fast
     */
    struct MinimaxHeap {
        int n;
        int k;
        int nvalid;

        std::vector<storage_idx_t> ids;
        std::vector<float> dis;
        typedef faiss::CMax<float, storage_idx_t> HC;

        explicit MinimaxHeap(int _n) : n(_n), k(0), nvalid(0), ids(_n), dis(_n) {}

        void push(storage_idx_t i, float v) {
            if (k == n) {
                if (v >= dis[0])
                    return;
                faiss::heap_pop<HC>(k--, dis.data(), ids.data());
                --nvalid;
            }
            faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
            ++nvalid;
        }

        float max() const {
            return dis[0];
        }

        int size() const {
            return nvalid;
        }

        void clear() {
            nvalid = k = 0;
        }

        int pop_min(float* vmin_out = nullptr) {
            assert(k > 0);
            // returns min. This is an O(n) operation
            int i = k - 1;
            while (i >= 0) {
                if (ids[i] != -1)
                    break;
                i--;
            }
            if (i == -1)
                return -1;
            int imin = i;
            float vmin = dis[i];
            i--;
            while (i >= 0) {
                if (ids[i] != -1 && dis[i] < vmin) {
                    vmin = dis[i];
                    imin = i;
                }
                i--;
            }
            if (vmin_out)
                *vmin_out = vmin;
            int ret = ids[imin];
            ids[imin] = -1;
            --nvalid;

            return ret;
        }

        int count_below(float thresh) {
            int n_below = 0;
            for (int i = 0; i < k; i++) {
                if (dis[i] < thresh) {
                    n_below++;
                }
            }

            return n_below;
        }
    };

    /// to sort pairs of (id, distance) from nearest to fathest or the reverse
    struct NodeDistCloser {
        float d;
        int id;
        NodeDistCloser(float _d, int _id) : d(_d), id(_id) {}
        bool operator<(const NodeDistCloser& obj1) const {
            return d < obj1.d;
        }
    };

    struct NodeDistFarther {
        float d;
        int id;
        NodeDistFarther(float _d, int _id) : d(_d), id(_id) {}
        bool operator<(const NodeDistFarther& obj1) const {
            return d > obj1.d;
        }
    };

    struct CompareByFirst {
        constexpr bool operator()(Node const& a, Node const& b) const noexcept {
            return a.first < b.first;
        }
    };

    /// level of each vector (base level = 1), size = ntotal
    std::vector<int> levels;

    /// entry point in the search structure (one of the points with maximum
    /// level
    storage_idx_t entry_point;

    faiss::RandomGenerator rng;
    std::default_random_engine level_generator;

    /// maximum level
    int max_level;
    int M;
    char* level0_links;
    char** linkLists;
    size_t level0_link_size;
    size_t link_size;
    double level_constant;
    VisitedListPool* visited_list_pool;

    bool has_deletion = false;

    bool loaded = false;

    /// expansion factor at construction time
    int efConstruction;

    /// expansion factor at search time
    int efSearch;

    /// range of entries in the neighbors table of vertex no at layer_no
    storage_idx_t* get_neighbor_link(idx_t no, int layer_no) const {
        return layer_no == 0
                ? reinterpret_cast<int*>(level0_links + no * level0_link_size)
                : reinterpret_cast<int*>(linkLists[no] + (layer_no - 1) * link_size);
    }
    unsigned short int get_neighbors_num(int* p) const {
        return *(reinterpret_cast<unsigned short int*>(p));
    }
    void set_neighbors_num(int* p, unsigned short int num) const {
        *(reinterpret_cast<unsigned short int*>(p)) = *(reinterpret_cast<unsigned short int*>(&num));
    }

    /// only mandatory parameter: nb of neighbors
    explicit HNSWfast(int M = 32);
    ~HNSWfast();

    void init(int ntotal);
    /// pick a random level for a new point, arg = 1/log(M)
    int random_level(double arg) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator)) * arg;
        return static_cast<int>(r);
    }

    void dump_level0(int current);

    void reset();

    int prepare_level_tab(size_t n, bool preset_levels = false);

    // re-implementations inspired by hnswlib
    /** add point pt_id on all levels <= pt_level and build the link
     * structure for them. inspired by implementation of hnswlib */
    void addPoint(DistanceComputer& ptdis, int pt_level, int pt_id);

    std::priority_queue<Node, std::vector<Node>, CompareByFirst> search_layer(
            DistanceComputer& ptdis,
            storage_idx_t nearest,
            int level);

    std::priority_queue<Node, std::vector<Node>, CompareByFirst>
    search_base_layer(
            DistanceComputer& ptdis,
            storage_idx_t nearest,
            storage_idx_t ef,
            float d_nearest,
            const SearchParametersHNSW* param) const;

    int make_connection(
            DistanceComputer& ptdis,
            storage_idx_t pt_id,
            std::priority_queue<Node, std::vector<Node>, CompareByFirst>& cand,
            int level);

    void prune_neighbors(
            DistanceComputer& ptdis,
            std::priority_queue<Node, std::vector<Node>, CompareByFirst>& cand,
            const int maxM,
            int* ret,
            int& ret_len);

    /// search interface inspired by hnswlib
    void searchKnn(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            const SearchParametersHNSW* param = nullptr) const;

    HNSWfast& operator=(const HNSWfast& /* rhs */) {
        return *this;
    }

    void init_link_list_lock(int ntotal) {
        std::vector<std::mutex>(ntotal).swap(link_list_locks);
    }

   private:
    std::vector<std::mutex> link_list_locks;
    std::mutex global;
};

/**************************************************************
 * Auxiliary structures
 **************************************************************/

typedef unsigned short int vl_type;

class VisitedList {
   public:
    vl_type curV;
    vl_type* mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    // keep compatibae with original version VisitedTable
    /// set flog #no to true
    void set(int no) {
        mass[no] = curV;
    }

    /// get flag #no
    bool get(int no) const {
        return mass[no] == curV;
    }

    void advance() {
        reset();
    }

    ~VisitedList() {
        delete[] mass;
    }
};

///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList*> pool;
    std::mutex poolguard;
    int numelements;

   public:
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
    }

    VisitedList* getFreeVisitedList() {
        VisitedList* rez;
        {
            std::unique_lock<std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList* vl) {
        std::unique_lock<std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList* rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }

    int64_t GetSize() {
        auto visit_list_size =
                sizeof(VisitedList) + numelements * sizeof(vl_type);
        auto pool_size = pool.size() * (sizeof(VisitedList*) + visit_list_size);
        return pool_size + sizeof(*this);
    }
};

} // namespace faiss
