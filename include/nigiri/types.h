#pragma once

#include <chrono>
#include <cinttypes>

#include "date/date.h"
#include "date/tz.h"

#include "ankerl/cista_adapter.h"

#include "cista/containers/bitset.h"
#include "cista/containers/bitvec.h"
#include "cista/containers/flat_matrix.h"
#include "cista/containers/mutable_fws_multimap.h"
#include "cista/containers/optional.h"
#include "cista/containers/string.h"
#include "cista/containers/tuple.h"
#include "cista/containers/variant.h"
#include "cista/containers/vector.h"
#include "cista/containers/vecvec.h"
#include "cista/reflection/printable.h"
#include "cista/strong.h"

namespace nigiri {

// Extend interval by one day. This is required due to the departure/arrival
// times being given in local time. After converting local times to UTC, this
// can result in times before the specified first day and after the specified
// last day (e.g. 00:30 CET is 23:30 UTC the day before, 22:30 PT is 05:30 UTC
// the next day). To be able to support this, the internal nigiri timetable
// range needs to start one day early and end one day longer than specified.
//
// There are trains that travel up to 4 days. Therefore, they are contained in
// the timetable even if their first departure is 4 days before the first day of
// the selected timetable period. To be able to fit those trains into the
// traffic day bitset, we prepend 4 in front of the timetable (in addition
// to the base day offset due to timezone conversion - see above).
constexpr auto const kTimetableOffset =
    std::chrono::days{1} + std::chrono::days{4};

template <size_t Size>
using bitset = cista::bitset<Size>;

constexpr auto const kMaxDays = 512;
using bitfield = bitset<kMaxDays>;

using bitvec = cista::offset::bitvec;

template <typename... Args>
using tuple = cista::tuple<Args...>;

template <typename A, typename B>
using pair = cista::pair<A, B>;

template <typename K, typename V>
using vector_map = cista::offset::vector_map<K, V>;

template <typename T>
using vector = cista::offset::vector<T>;

template <typename T>
using matrix = cista::raw::flat_matrix<T>;
using cista::raw::make_flat_matrix;

using cista::offset::to_vec;

template <typename... Ts>
using variant = cista::variant<Ts...>;
using cista::get;
using cista::holds_alternative;

template <typename K, typename V>
using vecvec = cista::offset::vecvec<K, V>;

template <typename K, typename V>
using mutable_fws_multimap = cista::offset::mutable_fws_multimap<K, V>;

template <typename K,
          typename V,
          typename Hash = cista::hash_all,
          typename Equality = cista::equals_all>
using hash_map = cista::offset::ankerl_map<K, V, Hash>;

template <typename K,
          typename Hash = cista::hash_all,
          typename Equality = cista::equals_all>
using hash_set = cista::offset::ankerl_set<K, Hash>;

using string = cista::offset::string;

template <typename T>
using optional = cista::optional<T>;

using bitfield_idx_t = cista::strong<std::uint32_t, struct _bitfield_idx>;
using location_idx_t = cista::strong<std::uint32_t, struct _location_idx>;
using osm_node_id_t = cista::strong<std::int64_t, struct _osm_node_idx>;
using route_idx_t = cista::strong<std::uint32_t, struct _location_idx>;
using section_idx_t = cista::strong<std::uint32_t, struct _section_idx>;
using section_db_idx_t = cista::strong<std::uint32_t, struct _section_db_idx>;
using trip_idx_t = cista::strong<std::uint32_t, struct _trip_idx>;
using trip_id_idx_t = cista::strong<std::uint32_t, struct _trip_id_str_idx>;
using transport_idx_t = cista::strong<std::uint32_t, struct _transport_idx>;
using rt_trip_idx_t = cista::strong<std::uint32_t, struct _rt_trip_idx>;
using source_idx_t = cista::strong<std::uint8_t, struct _source_idx>;
using day_idx_t = cista::strong<std::uint16_t, struct _day_idx>;
using timezone_idx_t = cista::strong<std::uint8_t, struct _timezone_idx>;
using merged_trips_idx_t =
    cista::strong<std::uint32_t, struct _merged_trips_idx>;
using footpath_idx_t = cista::strong<std::uint32_t, struct _footpath_idx>;
using source_file_idx_t = cista::strong<std::uint16_t, struct _source_file_idx>;
using track_name_idx_t = cista::strong<std::uint16_t, struct _track_name_idx>;

using line_id_t = string;

using trip_direction_string_idx_t =
    cista::strong<std::uint32_t, struct _trip_direction_string>;
using trip_direction_t =
    cista::variant<location_idx_t, trip_direction_string_idx_t>;
using trip_direction_idx_t =
    cista::strong<std::uint32_t, struct _trip_direction_idx>;
using trip_line_idx_t = cista::strong<std::uint32_t, struct _trip_line_idx>;
using attribute_idx_t = cista::strong<std::uint32_t, struct _attribute_idx>;
using attribute_combination_idx_t =
    cista::strong<std::uint32_t, struct _attribute_combination>;
using provider_idx_t = cista::strong<std::uint32_t, struct _provider_idx>;

struct trip_debug {
  source_file_idx_t source_file_idx_;
  std::uint32_t line_number_from_, line_number_to_;
};

struct attribute {
  CISTA_PRINTABLE(attribute, "code", "text")
  friend bool operator==(attribute const&, attribute const&) = default;
  string code_, text_;
};

struct provider {
  CISTA_COMPARABLE()
  CISTA_PRINTABLE(provider, "short_name", "long_name")
  string short_name_, long_name_;
  timezone_idx_t tz_{timezone_idx_t::invalid()};
};

struct trip_id {
  CISTA_COMPARABLE()
  CISTA_PRINTABLE(trip_id, "id", "src")
  std::string id_;
  source_idx_t src_;
};

struct location_id {
  CISTA_COMPARABLE()
  CISTA_PRINTABLE(location_id, "id", "src")
  string id_;
  source_idx_t src_;
};

struct transport {
  CISTA_PRINTABLE(transport, "idx", "day")
  constexpr bool is_valid() const { return day_ != day_idx_t::invalid(); }
  transport_idx_t t_idx_{transport_idx_t::invalid()};
  day_idx_t day_{day_idx_t::invalid()};
};

using i32_minutes = std::chrono::duration<std::int32_t, std::ratio<60>>;
using i16_minutes = std::chrono::duration<std::int16_t, std::ratio<60>>;
using u8_minutes = std::chrono::duration<std::uint8_t, std::ratio<60>>;
using duration_t = i16_minutes;
using unixtime_t = std::chrono::sys_time<i32_minutes>;
using local_time = date::local_time<i32_minutes>;

constexpr u8_minutes operator"" _i8_minutes(unsigned long long n) {
  return duration_t{n};
}

constexpr duration_t operator"" _minutes(unsigned long long n) {
  return duration_t{n};
}

constexpr duration_t operator"" _hours(unsigned long long n) {
  return duration_t{n * 60U};
}

constexpr duration_t operator"" _days(unsigned long long n) {
  return duration_t{n * 1440U};
}

using minutes_after_midnight_t = duration_t;

struct tz_offsets {
  struct season {
    duration_t offset_{0};
    unixtime_t begin_{unixtime_t::min()}, end_{unixtime_t::max()};
    duration_t season_begin_mam_{0};
    duration_t season_end_mam_{0};
  };
  friend std::ostream& operator<<(std::ostream&, tz_offsets const&);
  vector<season> seasons_;
  duration_t offset_{0};
};

using timezone = variant<void const*, tz_offsets>;

enum class clasz : std::uint8_t {
  kAir = 0,
  kHighSpeed = 1,
  kLongDistance = 2,
  kCoach = 3,
  kNight = 4,
  kRegionalFast = 5,
  kRegional = 6,
  kMetro = 7,
  kSubway = 8,
  kTram = 9,
  kBus = 10,
  kShip = 11,
  kOther = 12,
  kNumClasses
};

constexpr auto const kNumClasses =
    static_cast<std::underlying_type_t<clasz>>(clasz::kNumClasses);

enum class location_type : std::uint8_t { kTrack, kPlatform, kStation };

enum class event_type { kArr, kDep };
enum class direction { kForward, kBackward };

}  // namespace nigiri

#include <iomanip>
#include <ostream>

#include "cista/serialization.h"
#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"

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
                                nigiri::i32_minutes const& t) {
  auto const days = t.count() / 1440;
  auto const hours = (t.count() % 1440) / 60;
  auto const minutes = ((t.count() % 1440) % 60);
  return out << std::setw(2) << std::setfill('0') << hours << ':'  //
             << std::setw(2) << std::setfill('0') << minutes << '.' << days;
}

inline std::ostream& operator<<(std::ostream& out,
                                nigiri::unixtime_t const& t) {
  date::to_stream(out, "%F %R", t);
  return out;
}

}  // namespace std::chrono

#include <iostream>

namespace nigiri {

inline local_time to_local_time_offsets(tz_offsets const& offsets,
                                        unixtime_t const t) {
  auto const active_season_it =
      utl::find_if(offsets.seasons_, [&](tz_offsets::season const& s) {
        auto const season_begin =
            s.begin_ + s.season_begin_mam_ - offsets.offset_;
        auto const season_end = s.end_ + s.season_end_mam_ - s.offset_;
        return t >= season_begin && t < season_end;
      });
  auto const active_offset = active_season_it == end(offsets.seasons_)
                                 ? offsets.offset_
                                 : active_season_it->offset_;
  return local_time{(t + active_offset).time_since_epoch()};
}

inline local_time to_local_time_tz(date::time_zone const* tz,
                                   unixtime_t const t) {
  return std::chrono::time_point_cast<i32_minutes>(tz->to_local(t));
}

inline local_time to_local_time(timezone const& tz, unixtime_t const t) {
  return tz.apply(utl::overloaded{
      [t](tz_offsets const& x) { return to_local_time_offsets(x, t); },
      [t](void const* x) {
        return to_local_time_tz(reinterpret_cast<date::time_zone const*>(x), t);
      }});
}

}  // namespace nigiri
