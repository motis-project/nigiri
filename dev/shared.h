#pragma once
// Duplicated from libosmium and osr


#include <cstdint>
#include <cmath>
#include <ranges>

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

/* ---- Shared functions ---- */

using coordinate_type = int32_t;
using id_type = std::string;
using id_map_type = std::vector<id_type>;

constexpr std::string_view cache_file_template{"shape-cache.{}.dat"};
constexpr std::string_view id_map_file{"shape-id.dat"};  // Might be anything?


inline auto get_cache(cista::mmap::protection mode) {
  return mm_vecvec<std::size_t, coordinate_type>{
      cista::basic_mmap_vec<coordinate_type, std::size_t>{
          cista::mmap{std::format(cache_file_template, "values").data(), mode}},
      cista::basic_mmap_vec<std::size_t, std::size_t>{cista::mmap{
          std::format(cache_file_template, "metadata").data(), mode}}};
}

inline auto get_cache_writer() { return get_cache(cista::mmap::protection::WRITE); }

inline auto get_cache_reader() { return get_cache(cista::mmap::protection::READ); }


inline auto get_mapper(cista::mmap::protection mode) {
  return cista::mmap_vec<id_type::value_type>{cista::mmap{id_map_file.data(), mode}};
}

inline auto get_map_reader() { return get_mapper(cista::mmap::protection::READ); }

inline auto get_map_writer() { return get_mapper(cista::mmap::protection::WRITE); }


inline auto vec_to_map(id_map_type ids) {
    std::unordered_map<id_type, size_t> m;
    for (auto [pos, id] : std::ranges::enumerate_view(ids)) {
        m.insert({id, pos});
    }
    return m;
}