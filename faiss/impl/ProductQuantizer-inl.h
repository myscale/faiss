/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace faiss {

inline PQEncoderGeneric::PQEncoderGeneric(
        uint8_t* _code,
        int _nbits,
        uint8_t _offset)
        : code(_code), offset(_offset), nbits(_nbits), reg(0) {
    assert(_nbits <= 64);
    if (_offset > 0) {
        reg = (*_code & ((1 << _offset) - 1));
    }
}

inline void PQEncoderGeneric::encode(uint64_t x) {
    reg |= static_cast<uint8_t>(x << offset);
    x >>= (8 - offset);
    if (offset + nbits >= 8) {
        *code++ = reg;

        for (int i = 0; i < (nbits - (8 - offset)) / 8; ++i) {
            *code++ = static_cast<uint8_t>(x);
            x >>= 8;
        }

        offset += nbits;
        offset &= 7;
        reg = static_cast<uint8_t>(x);
    } else {
        offset += nbits;
    }
}

inline PQEncoderGeneric::~PQEncoderGeneric() {
    if (offset > 0) {
        *code = reg;
    }
}

inline PQEncoder8::PQEncoder8(uint8_t* _code, int nbits) : code(_code) {
    (void)nbits;
    assert(8 == nbits);
}

inline void PQEncoder8::encode(uint64_t x) {
    *code++ = static_cast<uint8_t>(x);
}

inline PQEncoder16::PQEncoder16(uint8_t* _code, int nbits)
        : code(reinterpret_cast<uint16_t*>(_code)) {
    (void)nbits;
    assert(16 == nbits);
}

inline void PQEncoder16::encode(uint64_t x) {
    *code++ = static_cast<uint16_t>(x);
}

inline PQDecoderGeneric::PQDecoderGeneric(const uint8_t* _code, int _nbits)
        : code(_code),
          offset(0),
          nbits(_nbits),
          mask((1ull << _nbits) - 1),
          reg(0) {
    assert(_nbits <= 64);
}

inline uint64_t PQDecoderGeneric::decode() {
    if (offset == 0) {
        reg = *code;
    }
    uint64_t c = (reg >> offset);

    if (offset + nbits >= 8) {
        uint64_t e = 8 - offset;
        ++code;
        for (int i = 0; i < (nbits - (8 - offset)) / 8; ++i) {
            c |= (static_cast<uint64_t>(*code++) << e);
            e += 8;
        }

        offset += nbits;
        offset &= 7;
        if (offset > 0) {
            reg = *code;
            c |= (static_cast<uint64_t>(reg) << e);
        }
    } else {
        offset += nbits;
    }

    return c & mask;
}

inline PQDecoder8::PQDecoder8(const uint8_t* _code, int nbits_in) : code(_code) {
    (void)nbits_in;
    assert(8 == nbits_in);
}

inline uint64_t PQDecoder8::decode() {
    return static_cast<uint64_t>(*code++);
}

inline PQDecoder16::PQDecoder16(const uint8_t* _code, int nbits_in)
        : code(reinterpret_cast<const uint16_t*>(_code)) {
    (void)nbits_in;
    assert(16 == nbits_in);
}

inline uint64_t PQDecoder16::decode() {
    return static_cast<uint64_t>(*code++);
}

} // namespace faiss
