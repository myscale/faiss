#ifndef JACCARD_H
#define JACCARD_H
#include "jaccard-inl.h"
#include "hamming-inl.h"
#include <set>

/* Return closest neighbors w.r.t Hamming distance, using max count. */
template <class JaccardComputer>
static void jaccard_knn_mc(
        int bytes_per_code,
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        size_t k,
        float* distances,
        int64_t* labels,
        const faiss::IDSelector* sel) {
    struct IdDis {
        int64_t id;
        float distance;
        std::string as_string() {
            return "id: " + std::to_string(id) + ", distance: " + std::to_string(distance);
        }
    };
    auto comp = [](const IdDis& p1, const IdDis& p2) {
        return p1.distance < p2.distance;
    };
    for(int64_t i=0; i<na; i++){
        faiss::JaccardComputerDefault computer(
                a + bytes_per_code * i, bytes_per_code);
        std::multiset<IdDis, decltype(comp)> potential;
        for(int64_t j=0; j<nb; j++){
            if (sel && !sel->is_member(j)) {
                continue;
            }
            IdDis id_dist = {
                .id = j,
                .distance = computer.jaccard(b + bytes_per_code * j)
            };
            if(potential.size() < k) {
                potential.insert(id_dist);
                continue;
            }
           
            auto max_in_potential = *potential.rbegin();
            if (max_in_potential.distance > id_dist.distance) {
                potential.erase(std::prev(potential.end()));
                potential.insert(id_dist);
            }
        }
        int nres = 0;
        for(const auto & potential_item: potential) {
            labels[i * k + nres] = potential_item.id;
            distances[i * k + nres] = potential_item.distance;
            ++nres;
        }
        while(nres < k) {
            labels[i * k + nres] = -1;
            distances[i * k + nres] = -1;
            ++nres;
        }
    }
}

static void jaccard_knn(
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        size_t k,
        size_t ncodes,
        float* distances,
        int64_t* labels,
        const faiss::IDSelector* sel = nullptr) {
    jaccard_knn_mc<faiss::JaccardComputerDefault>(
            ncodes, a, b, na, nb, k, distances, labels, sel);
}

#endif
