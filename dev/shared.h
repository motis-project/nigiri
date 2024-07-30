#pragma once
// Duplicated from libosmium and osr


#include <cstdint>
#include <cmath>

#include "cista/containers/mmap_vec.h"
#include "cista/containers/vecvec.h"


template <typename T>
using mm_vec = cista::basic_mmap_vec<T, std::uint64_t>;

template <typename K, typename V, typename SizeType = cista::base_t<K>>
using mm_vecvec = cista::basic_vecvec<K, mm_vec<V>, mm_vec<SizeType>>;


constexpr int32_t coordinate_precision{10000000};

constexpr int32_t double_to_fix(const double c) noexcept {
    return static_cast<int32_t>(std::round(c * coordinate_precision));
}

constexpr double fix_to_double(const int32_t c) noexcept {
    return static_cast<double>(c) / coordinate_precision;
}