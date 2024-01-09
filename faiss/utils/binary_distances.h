// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License

#ifndef FAISS_BINARY_DISTANCE_H
#define FAISS_BINARY_DISTANCE_H

#include <faiss/MetricType.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/Heap.h>
#include <stdint.h>

/* The binary distance type */
typedef float tadis_t;

namespace faiss {

struct IDSelector;

/**
 * Calculate Jaccard distance
 */
static float bvec_jaccard(
        const uint8_t* data1,
        const uint8_t* data2,
        const size_t code_size) {
    // todo aguzhva: improve this code, maybe reuse the code from hamming.h
    int accu_num = 0, accu_den = 0;
    for(int i=0; i<code_size; i++){
        accu_num += std::popcount(
                static_cast<std::uint8_t>(data1[i] & data2[i]));
        accu_den += std::popcount(
                static_cast<std::uint8_t>(data1[i] | data2[i]));
    }
    return (accu_den == 0)
            ? 1.0
            : (static_cast<float>(accu_den - accu_num) / static_cast<float>(accu_den));
}


void binary_knn_mc(
        MetricType metric_type,
        const uint8_t* a,
        const uint8_t* b,
        size_t na,
        size_t nb,
        size_t k,
        size_t ncodes,
        float* distances,
        int64_t* labels,
        const IDSelector* sel = nullptr);

} // namespace faiss

#endif // FAISS_BINARY_DISTANCE_H
