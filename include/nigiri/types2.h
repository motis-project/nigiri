
#pragma once

#include "cista/containers/mmap_vec.h"
#include "cista/containers/vecvec.h"

template <typename T>
using mm_vec = cista::basic_mmap_vec<T, std::uint64_t>;

template <typename Key, typename V, typename SizeType = cista::base_t<Key>>
using mm_vecvec = cista::basic_vecvec<Key, mm_vec<V>, mm_vec<SizeType>>;
