#pragma once

#include <chrono>
#include <cinttypes>

#include "cista/containers/bitset.h"
#include "cista/containers/flat_matrix.h"
#include "cista/containers/fws_multimap.h"
#include "cista/containers/hash_map.h"
#include "cista/containers/hash_set.h"
#include "cista/containers/mutable_fws_multimap.h"
#include "cista/containers/string.h"
#include "cista/containers/tuple.h"
#include "cista/containers/variant.h"
#include "cista/containers/vector.h"
#include "cista/reflection/printable.h"
#include "cista/strong.h"

namespace nigiri {

using bitfield = cista::bitset<512>;

template <typename T>
using flat_matrix = cista::offset::flat_matrix<T>;

using cista::offset::make_flat_matrix;

template <typename... Args>
using tuple = cista::tuple<Args...>;

template <typename A, typename B>
using pair = cista::pair<A, B>;

template <typename K, typename V>
using vector_map = cista::offset::vector_map<K, V>;

template <typename T>
using vector = cista::offset::vector<T>;

using cista::offset::to_vec;

template <typename... Ts>
using variant = cista::variant<Ts...>;
using cista::get;
using cista::holds_alternative;

template <typename K, typename V>
using fws_multimap = cista::offset::fws_multimap<K, V>;

template <typename K, typename V>
using mutable_fws_multimap = cista::offset::mutable_fws_multimap<K, V>;

template <typename K, typename V, typename Hash = cista::hashing<K>>
using hash_map = cista::offset::hash_map<K, V, Hash>;

template <typename K, typename Hash = cista::hashing<K>>
using hash_set = cista::offset::hash_set<K, Hash>;

using string = cista::offset::string;

using bitfield_idx_t = cista::strong<std::uint32_t, struct _bitfield_idx>;
using location_idx_t = cista::strong<std::uint32_t, struct _location_idx>;
using osm_node_id_t = cista::strong<std::int64_t, struct _osm_node_idx>;
using route_idx_t = cista::strong<std::uint32_t, struct _location_idx>;
using section_idx_t = cista::strong<std::uint32_t, struct _section_idx>;
using section_db_idx_t = cista::strong<std::uint32_t, struct _section_db_idx>;
using external_trip_idx_t = cista::strong<std::uint32_t, struct _trip_idx>;
using trip_idx_t = cista::strong<std::uint32_t, struct _trip_idx>;
using rt_trip_idx_t = cista::strong<std::uint32_t, struct _rt_trip_idx>;
using source_idx_t = cista::strong<std::uint8_t, struct _source_idx>;
using equivalence_set_idx_t =
    cista::strong<std::uint32_t, struct _equivalence_set_idx>;
using merged_trips_idx_t =
    cista::strong<std::uint32_t, struct _merged_trips_idx>;
using output_rule_t = cista::strong<std::uint8_t, struct _output_rule>;

struct trip_id {
  CISTA_PRINTABLE(trip_id, "id", "src")
  string id_;
  source_idx_t src_;
};

struct location_id {
  CISTA_PRINTABLE(location_id, "id", "src")
  string id_;
  source_idx_t src_;
};

using duration_t = std::chrono::duration<std::uint16_t, std::ratio<60>>;
using unixtime_t = std::chrono::time_point<
    std::chrono::system_clock,
    std::chrono::duration<std::uint32_t, std::ratio<60>>>;

using minutes_after_midnight_t = duration_t;

enum class location_type : std::uint8_t {
  track,
  platform,
  station,
  meta_station
};

enum class event_type { ARR, DEP };

constexpr auto const kNoSource = source_idx_t{0};

}  // namespace nigiri

#include <iomanip>
#include <ostream>

#include "cista/serialization.h"

namespace std::chrono {

template <typename Ctx>
inline void serialize(Ctx& c,
                      nigiri::duration_t const* origin,
                      cista::offset_t pos) {
  c.write(pos, cista::convert_endian<Ctx::MODE>(origin->count()));
}

template <typename Ctx>
void deserialize(Ctx const& c, nigiri::duration_t* el) {
  c.convert_endian(*reinterpret_cast<nigiri::duration_t::rep*>(el));
  *el = nigiri::duration_t{*reinterpret_cast<nigiri::duration_t::rep*>(el)};
}

inline std::ostream& operator<<(std::ostream& out,
                                nigiri::duration_t const& t) {
  auto const days = t.count() / 1440;
  auto const hours = (t.count() % 1440) / 60;
  auto const minutes = ((t.count() % 1440) % 60);
  return out << std::setw(2) << std::setfill('0') << hours << ':'  //
             << std::setw(2) << std::setfill('0') << minutes << '.' << days;
}

inline std::ostream& operator<<(std::ostream& out,
                                nigiri::unixtime_t const& t) {
  auto const time = std::chrono::system_clock::to_time_t(t);
  auto const* tm = std::localtime(&time);
  char buffer[25];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm);
  return out << buffer;
}

inline std::ostream& operator<<(std::ostream& out, sys_days const& t) {
  auto const ymd = std::chrono::year_month_day{t};
  return out << static_cast<int>(ymd.year()) << '/' << std::setw(2)
             << std::setfill('0') << static_cast<unsigned>(ymd.month()) << '/'
             << std::setw(2) << static_cast<unsigned>(ymd.day());
}

}  // namespace std::chrono
