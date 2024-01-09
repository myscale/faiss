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

#ifndef FAISS_JACCARD_INL_H
#define FAISS_JACCARD_INL_H

#include "binary_distances.h"

namespace faiss {

struct JaccardComputerDefault {
    const uint8_t* a;
    int n;

    JaccardComputerDefault() {}

    JaccardComputerDefault(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        a = a8;
        n = code_size;
    }

    float jaccard(const uint8_t* b8) const {
        return bvec_jaccard(a, b8, n);
    }
};

} // namespace faiss

#endif
