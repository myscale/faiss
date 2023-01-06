#include "HNSWfast.h"

#include <string>
#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>

#include <iostream>

namespace faiss {

/**************************************************************
 * hnsw structure implementation
 **************************************************************/

HNSWfast::HNSWfast(int M) : M(M), rng(100) {
    level_generator.seed(100);
    max_level = -1;
    entry_point = -1;
    efSearch = 50;
    efConstruction = 70;
    level0_link_size = sizeof(int) * ((M << 1) | 1);
    link_size = sizeof(int) * (M + 1);
    level0_links = nullptr;
    linkLists = nullptr;
    level_constant = 1 / log(1.0 * M);
    visited_list_pool = nullptr;
}

void HNSWfast::init(int ntotal) {
    level_generator.seed(100);
    if (visited_list_pool)
        delete visited_list_pool;
    visited_list_pool = new VisitedListPool(1, ntotal);
    std::vector<std::mutex>(ntotal).swap(link_list_locks);

    levels.resize(ntotal);
    for (int i = 0; i < ntotal; i++) {
        int pt_level = random_level(level_constant);
        levels[i] = pt_level;
    }

    level0_links = (char*)malloc(level0_link_size * ntotal);
    if (level0_links == nullptr)
        throw std::runtime_error("No enough memory 4 level0_links!");
    memset(level0_links, 0, ntotal * level0_link_size);

    linkLists = (char**)malloc(sizeof(void*) * ntotal);
    if (linkLists == nullptr)
        throw std::runtime_error("No enough memory 4 level0_links_new!");
    memset(linkLists, 0, ntotal * sizeof(void*));

    for (int i = 0; i < ntotal; i++) {
        int pt_level = levels[i];
        if (pt_level > max_level)
            max_level = pt_level;
        if (pt_level) {
            linkLists[i] = (char*)malloc(link_size * pt_level);
            if (linkLists[i] == nullptr) {
                throw std::runtime_error("No enough memory 4 linkLists!");
            }
            memset(linkLists[i], 0, link_size * pt_level);
        }
    }
}

HNSWfast::~HNSWfast() {
    for (auto i = 0; i < levels.size(); ++i) {
        if (levels[i])
            free(linkLists[i]);
    }
    free(level0_links);
    free(linkLists);
    delete visited_list_pool;
}

void HNSWfast::reset() {
    max_level = -1;
    entry_point = -1;
    free(level0_links);
    for (auto i = 0; i < levels.size(); ++i) {
        if (levels[i])
            free(linkLists[i]);
    }
    free(linkLists);
    levels.clear();
    level0_links = nullptr;
    linkLists = nullptr;
    level_constant = 1 / log(1.0 * M);
}

int HNSWfast::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = levels.size();
    size_t n1 = n0 + n;

    if (preset_levels) {
        FAISS_ASSERT(n1 == levels.size());
    } else {
        levels.resize(n1);
        for (int i = 0; i < n; i++) {
            int pt_level = random_level(level_constant);
            levels[n0 + i] = pt_level;
        }
    }

    char* level0_links_new =
            (char*)realloc(level0_links, level0_link_size * n1);
    if (level0_links_new == nullptr)
        throw std::runtime_error("No enough memory 4 level0_links!");
    level0_links = level0_links_new;
    memset(level0_links + n0 * level0_link_size, 0, n * level0_link_size);

    char** linkLists_new = (char**)realloc(linkLists, sizeof(void*) * n1);
    if (linkLists_new == nullptr)
        throw std::runtime_error("No enough memory 4 level0_links_new!");
    linkLists = linkLists_new;
    memset(linkLists + n0 * sizeof(void*), 0, n * sizeof(void*));

    int debug_space = 0;
    for (int i = 0; i < n; i++) {
        int pt_level = levels[i + n0];
        if (pt_level > max_level)
            max_level = pt_level;
        if (pt_level) {
            linkLists[n0 + i] = (char*)malloc(link_size * pt_level);
            if (linkLists[n0 + i] == nullptr) {
                throw std::runtime_error("No enough memory 4 linkLists!");
            }
            memset(linkLists[n0 + i], 0, link_size * pt_level);
        }
    }

    //    std::vector<std::mutex>(n0 + n).swap(link_list_locks);
    //    if (visited_list_pool) delete visited_list_pool;
    //    visited_list_pool = new VisitedListPool(1, n1);

    return max_level;
}

void HNSWfast::dump_level0(int current) {
    std::cout << "hnswfast: " << __func__ << ": " << current << std::endl;
    for (int i = 0; i <= current; i++) {
        std::cout << "\t" << i << ":";
        int* curObj_link = get_neighbor_link(i, 0);
        auto curObj_nei_num = get_neighbors_num(curObj_link);
        for (int j = 0; j < curObj_nei_num; j++) {
            std::cout << curObj_link[j + 1] << ", ";
        }
        std::cout << std::endl;
    }
}

/**************************************************************
 * new implementation of hnsw ispired by hnswlib
 * by cmli@zilliz   July 30, 2020
 **************************************************************/
using Node = faiss::HNSWfast::Node;
using CompareByFirst = faiss::HNSWfast::CompareByFirst;
void HNSWfast::addPoint(DistanceComputer& ptdis, int pt_level, int pt_id) {
    std::unique_lock<std::mutex> lock_el(link_list_locks[pt_id]);
    std::unique_lock<std::mutex> temp_lock(global);
    int maxlevel_copy = max_level;
    if (pt_level <= maxlevel_copy)
        temp_lock.unlock();
    int currObj = entry_point;

    if (currObj != -1) {
        if (pt_level < maxlevel_copy) {
            float curdist = ptdis(currObj);
            for (int lev = maxlevel_copy; lev > pt_level; lev--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::unique_lock<std::mutex> lk(link_list_locks[currObj]);
                    int* curObj_link = get_neighbor_link(currObj, lev);
                    auto curObj_nei_num = get_neighbors_num(curObj_link);
                    for (auto i = 1; i <= curObj_nei_num; ++i) {
                        int cand = curObj_link[i];
                        if (cand < 0 || cand > levels.size())
                            throw std::runtime_error(
                                    "cand error when addPoint");
                        float d = ptdis(cand);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        for (int lev = std::min(pt_level, maxlevel_copy); lev >= 0; --lev) {
            if (lev > maxlevel_copy || lev < 0)
                throw std::runtime_error("Level error");

            std::priority_queue<Node, std::vector<Node>, CompareByFirst>
                    top_candidates = search_layer(ptdis, currObj, lev);
            currObj = make_connection(ptdis, pt_id, top_candidates, lev);
        }
    } else {
        entry_point = 0;
        max_level = pt_level;
    }

    if (pt_level > maxlevel_copy) {
        entry_point = pt_id;
        max_level = pt_level;
    }
    // dump_level0(pt_id);
}

std::priority_queue<Node, std::vector<Node>, CompareByFirst> HNSWfast::
        search_layer(
                DistanceComputer& ptdis,
                storage_idx_t nearest,
                int level) {
    VisitedList* vl = visited_list_pool->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<Node, std::vector<Node>, CompareByFirst> top_candidates;
    std::priority_queue<Node, std::vector<Node>, CompareByFirst> candidate_set;

    float d_nearest = ptdis(nearest);
    float lb = d_nearest;
    top_candidates.emplace(d_nearest, nearest);
    candidate_set.emplace(-d_nearest, nearest);
    visited_array[nearest] = visited_array_tag;

    while (!candidate_set.empty()) {
        Node currNode = candidate_set.top();
        if ((-currNode.first) > lb && top_candidates.size() == efConstruction)
            break;
        candidate_set.pop();
        int cur_id = currNode.second;
        std::unique_lock<std::mutex> lk(link_list_locks[cur_id]);
        int* cur_link = get_neighbor_link(cur_id, level);
        auto cur_neighbor_num = get_neighbors_num(cur_link);

        for (auto i = 1; i <= cur_neighbor_num; ++i) {
            int candidate_id = cur_link[i];
            if (visited_array[candidate_id] == visited_array_tag)
                continue;
            visited_array[candidate_id] = visited_array_tag;
            float dcand = ptdis(candidate_id);
            if (top_candidates.size() < efConstruction || lb > dcand) {
                candidate_set.emplace(-dcand, candidate_id);
                top_candidates.emplace(dcand, candidate_id);
                if (top_candidates.size() > efConstruction)
                    top_candidates.pop();
                if (!top_candidates.empty())
                    lb = top_candidates.top().first;
            }
        }
    }
    visited_list_pool->releaseVisitedList(vl);
    return top_candidates;
}

std::priority_queue<Node, std::vector<Node>, CompareByFirst> HNSWfast::
        search_base_layer(
                DistanceComputer& ptdis,
                storage_idx_t nearest,
                storage_idx_t ef,
                float d_nearest,
                const SearchParametersHNSW* param) const {
    const IDSelector* sel = param ? param->sel : nullptr;
    VisitedList* vl = visited_list_pool->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<Node, std::vector<Node>, CompareByFirst> top_candidates;
    std::priority_queue<Node, std::vector<Node>, CompareByFirst> candidate_set;

    float lb;
    if (!sel || sel->is_member(nearest)) {
        lb = d_nearest;
        top_candidates.emplace(d_nearest, nearest);
        candidate_set.emplace(-d_nearest, nearest);
    } else {
        lb = std::numeric_limits<float>::max();
        candidate_set.emplace(-lb, nearest);
    }
    visited_array[nearest] = visited_array_tag;

    while (!candidate_set.empty()) {
        Node currNode = candidate_set.top();
        if ((-currNode.first) > lb && top_candidates.size() == ef)
            break;
        candidate_set.pop();
        int cur_id = currNode.second;
        int* cur_link = get_neighbor_link(cur_id, 0);
        auto cur_neighbor_num = get_neighbors_num(cur_link);
        for (auto i = 1; i <= cur_neighbor_num; ++i) {
            int candidate_id = cur_link[i];
            if (visited_array[candidate_id] != visited_array_tag) {
                visited_array[candidate_id] = visited_array_tag;
                float dcand = ptdis(candidate_id);
                if (top_candidates.size() < ef || lb > dcand) {
                    candidate_set.emplace(-dcand, candidate_id);
                    // check if candidate_id is enabled
                    if ((!sel || sel->is_member(candidate_id)))
                        top_candidates.emplace(dcand, candidate_id);
                    if (top_candidates.size() > ef)
                        top_candidates.pop();
                    if (!top_candidates.empty())
                        lb = top_candidates.top().first;
                }
            }
        }
    }
    //std::cout << "distance_compute: " << distance_compute << ", "
    //          << "ef: " << ef << ", M: " << M
    //          << std::endl;
    visited_list_pool->releaseVisitedList(vl);
    return top_candidates;
}

int HNSWfast::make_connection(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        std::priority_queue<Node, std::vector<Node>, CompareByFirst>& cand,
        int level) {
    int maxM = level ? M : M << 1;
    int* selectedNeighbors = (int*)malloc(sizeof(int) * maxM);
    int selectedNeighborsNum = 0;
    prune_neighbors(ptdis, cand, M, selectedNeighbors, selectedNeighborsNum);
    if (selectedNeighborsNum > M)
        throw std::runtime_error(
                "Wrong size of candidates returned by prune_neighbors!");

    int next_closest_entry_point = selectedNeighbors[0];

    int* cur_link = get_neighbor_link(pt_id, level);
    if (*cur_link)
        throw std::runtime_error(
                "The newly inserted element should have blank link");

    set_neighbors_num(cur_link, selectedNeighborsNum);
    for (auto i = 1; i <= selectedNeighborsNum; ++i) {
        if (cur_link[i])
            throw std::runtime_error("Possible memory corruption.");
        if (level > levels[selectedNeighbors[i - 1]])
            throw std::runtime_error(
                    "Trying to make a link on a non-exisitent level.");
        cur_link[i] = selectedNeighbors[i - 1];
    }

    for (auto i = 0; i < selectedNeighborsNum; ++i) {
        std::unique_lock<std::mutex> lk(link_list_locks[selectedNeighbors[i]]);

        int* selected_link = get_neighbor_link(selectedNeighbors[i], level);
        auto selected_neighbor_num = get_neighbors_num(selected_link);
        if (selected_neighbor_num > maxM)
            throw std::runtime_error("Bad value of selected_neighbor_num.");
        if (selectedNeighbors[i] == pt_id)
            throw std::runtime_error("Trying to connect an element to itself.");
        if (level > levels[selectedNeighbors[i]])
            throw std::runtime_error(
                    "Trying to make a link on a non-exisitent level.");
        if (selected_neighbor_num < maxM) {
            selected_link[selected_neighbor_num + 1] = pt_id;
            set_neighbors_num(selected_link, selected_neighbor_num + 1);
        } else {
            double d_max = ptdis(selectedNeighbors[i]);
            std::priority_queue<Node, std::vector<Node>, CompareByFirst> candi;
            candi.emplace(d_max, pt_id);
            for (auto j = 1; j <= selected_neighbor_num; ++j)
                candi.emplace(
                        ptdis.symmetric_dis(
                                selectedNeighbors[i], selected_link[j]),
                        selected_link[j]);
            int indx = 0;
            prune_neighbors(ptdis, candi, maxM, selected_link + 1, indx);
            set_neighbors_num(selected_link, indx);
        }
    }

    free(selectedNeighbors);
    return next_closest_entry_point;
}

void HNSWfast::prune_neighbors(
        DistanceComputer& ptdis,
        std::priority_queue<Node, std::vector<Node>, CompareByFirst>& cand,
        const int maxM,
        int* ret,
        int& ret_len) {
    if (cand.size() < maxM) {
        ret_len = cand.size();
        for (int i = static_cast<int>(cand.size()) - 1; i >= 0; i--) {
            ret[i] = cand.top().second;
            cand.pop();
        }
    } else if (maxM > 0) {
        ret_len = 0;

        std::vector<Node> queue_closest;
        queue_closest.resize(cand.size());
        for (int i = static_cast<int>(cand.size()) - 1; i >= 0; i--) {
            queue_closest[i] = cand.top();
            cand.pop();
        }

        for (auto& curr : queue_closest) {
            bool good = true;
            for (auto i = 0; i < ret_len; ++i) {
                float cur_dist = ptdis.symmetric_dis(curr.second, ret[i]);
                if (cur_dist < curr.first) {
                    good = false;
                    break;
                }
            }
            if (good) {
                ret[ret_len++] = (curr.second);
                if (ret_len >= maxM) {
                    break;
                }
            }
        }
    }
}

void HNSWfast::searchKnn(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        const SearchParametersHNSW* param) const {
    if (levels.size() == 0)
        return;
    int ep = entry_point;
    float dist = qdis(ep);

    for (auto i = max_level; i > 0; --i) {
        bool good = true;
        while (good) {
            good = false;
            int* ep_link = get_neighbor_link(ep, i);
            auto ep_neighbors_cnt = get_neighbors_num(ep_link);
            for (auto j = 1; j <= ep_neighbors_cnt; ++j) {
                int cand = ep_link[j];
                if (cand < 0 || cand > levels.size())
                    throw std::runtime_error("cand error");
                float d = qdis(cand);
                if (d < dist) {
                    dist = d;
                    ep = cand;
                    good = true;
                }
            }
        }
    }
    int ef_search = efSearch;
    if (param) {
        ef_search = param->efSearch;
    }
    std::priority_queue<Node, std::vector<Node>, CompareByFirst>
            top_candidates = search_base_layer(
                    qdis, ep, std::max(ef_search, k), dist, param);
    while (top_candidates.size() > k)
        top_candidates.pop();
    int rst_num = top_candidates.size();
    int i = rst_num - 1;
    while (!top_candidates.empty()) {
        I[i] = top_candidates.top().second;
        D[i] = top_candidates.top().first;
        i--;
        top_candidates.pop();
    }
    for (; rst_num < k; rst_num++) {
        I[rst_num] = -1;
        D[rst_num] = 1.0 / 0.0;
    }
}

} // namespace faiss
