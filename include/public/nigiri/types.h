#pragma once

#include <chrono>
#include <cinttypes>

#include "cista/containers/bitset.h"
#include "cista/containers/fws_multimap.h"
#include "cista/containers/hash_map.h"
#include "cista/containers/string.h"
#include "cista/containers/tuple.h"
#include "cista/containers/variant.h"
#include "cista/containers/vector.h"
#include "cista/strong.h"

namespace nigiri {

using bitfield = cista::bitset<512>;

template <typename... Args>
using tuple = cista::tuple<Args...>;

template <typename A, typename B>
using pair = cista::pair<A, B>;

template <typename K, typename V>
using vector_map = cista::offset::vector_map<K, V>;

template <typename T>
using vector = cista::offset::vector<T>;

template <typename K, typename V>
using fws_multimap = cista::offset::fws_multimap<K, V>;

template <typename K, typename V>
using hash_map = cista::offset::hash_map<K, V>;

using string = cista::offset::string;
using bitfield_idx_t = cista::strong<std::uint32_t, struct _bitfield_idx>;
using location_idx_t = cista::strong<std::uint32_t, struct _location_idx>;
using route_idx_t = cista::strong<std::uint32_t, struct _location_idx>;
using section_idx_t = cista::strong<std::uint32_t, struct _section_idx>;
using section_db_idx_t = cista::strong<std::uint32_t, struct _section_db_idx>;
using external_trip_idx_t = cista::strong<std::uint32_t, struct _trip_idx>;
using trip_idx_t = cista::strong<std::uint32_t, struct _trip_idx>;
using external_trip_id_t = cista::strong<string, struct _trip_id>;
using location_id_t = cista::strong<string, struct _station_id>;

using duration_t = std::chrono::duration<std::uint16_t, std::ratio<60>>;
using unixtime_t = std::chrono::time_point<
    std::chrono::system_clock,
    std::chrono::duration<std::uint32_t, std::ratio<60>>>;

using minutes_after_midnight_t = duration_t;

}  // namespace nigiri
