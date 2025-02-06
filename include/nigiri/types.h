#pragma once

#include <chrono>
#include <cinttypes>
#include <variant>

#include "fmt/ostream.h"

#include "date/date.h"
#include "date/tz.h"

#include "ankerl/cista_adapter.h"

#include "cista/containers/array.h"
#include "cista/containers/bitset.h"
#include "cista/containers/bitvec.h"
#include "cista/containers/flat_matrix.h"
#include "cista/containers/mutable_fws_multimap.h"
#include "cista/containers/nvec.h"
#include "cista/containers/optional.h"
#include "cista/containers/string.h"
#include "cista/containers/tuple.h"
#include "cista/containers/variant.h"
#include "cista/containers/vector.h"
#include "cista/containers/vecvec.h"
#include "cista/reflection/printable.h"
#include "cista/strong.h"

#include "geo/latlng.h"

#include "nigiri/common/interval.h"
#include "nigiri/common/it_range.h"

#include <tg.h>

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

using bitvec = cista::raw::bitvec;

template <typename K = std::uint32_t>
using bitvec_map = cista::basic_bitvec<cista::raw::vector<std::uint64_t>, K>;

template <typename... Args>
using tuple = cista::tuple<Args...>;

template <typename A, typename B>
using pair = cista::pair<A, B>;

template <typename K, typename V>
using vector_map = cista::raw::vector_map<K, V>;

template <typename V, std::size_t SIZE>
using array = cista::raw::array<V, SIZE>;

template <typename T>
using vector = cista::raw::vector<T>;

template <typename T>
using matrix = cista::raw::flat_matrix<T>;
using cista::raw::make_flat_matrix;

using cista::raw::to_vec;

template <typename... Ts>
using variant = cista::variant<Ts...>;
using cista::get;
using cista::holds_alternative;

template <typename K, typename V, typename SizeType = cista::base_t<K>>
using vecvec = cista::raw::vecvec<K, V, SizeType>;

template <typename K, typename V>
using mutable_fws_multimap = cista::raw::mutable_fws_multimap<K, V>;

template <typename K,
          typename V,
          typename Hash = cista::hash_all,
          typename Equality = cista::equals_all>
using hash_map = cista::raw::ankerl_map<K, V, Hash, Equality>;

template <typename K,
          typename Hash = cista::hash_all,
          typename Equality = cista::equals_all>
using hash_set = cista::raw::ankerl_set<K, Hash, Equality>;

using stop_idx_t = std::uint16_t;

using string = cista::raw::string;

template <typename T>
using optional = cista::optional<T>;

template <typename Key, typename T, std::size_t N>
using nvec = cista::raw::nvec<Key, T, N>;

template <typename K, typename V>
using mm_vec_map = cista::basic_mmap_vec<V, K>;

template <typename T>
using mm_vec = cista::basic_mmap_vec<T, std::uint64_t>;

template <typename Key, typename V, typename SizeType = cista::base_t<Key>>
using mm_vecvec = cista::basic_vecvec<Key, mm_vec<V>, mm_vec<SizeType>>;

template <typename Key, typename T>
struct paged_vecvec_helper {
  using data_t =
      cista::paged<vector<T>, std::uint64_t, std::uint32_t, 4U, 1U << 31U>;
  using idx_t = vector<typename data_t::page_t>;
  using type = cista::paged_vecvec<idx_t, data_t, Key>;
};

template <typename Key, typename T>
using paged_vecvec = paged_vecvec_helper<Key, T>::type;

template <typename Key, typename T>
struct mm_paged_vecvec_helper {
  using data_t =
      cista::paged<mm_vec<T>, std::uint64_t, std::uint32_t, 2U, 1U << 31U>;
  using idx_t = mm_vec<typename data_t::page_t>;
  using type = cista::paged_vecvec<idx_t, data_t, Key>;
};

template <typename Key, typename T>
using mm_paged_vecvec = mm_paged_vecvec_helper<Key, T>::type;

using bitfield_idx_t = cista::strong<std::uint32_t, struct _bitfield_idx>;
using location_idx_t = cista::strong<std::uint32_t, struct _location_idx>;
using geometry_idx_t = cista::strong<std::uint32_t, struct _geometry_idx>;
using area_idx_t = cista::strong<std::uint32_t, struct _area_idx>;
using area_element_idx_t = cista::strong<std::uint32_t, struct _area_idx>;
using route_idx_t = cista::strong<std::uint32_t, struct _route_idx>;
using section_idx_t = cista::strong<std::uint32_t, struct _section_idx>;
using section_db_idx_t = cista::strong<std::uint32_t, struct _section_db_idx>;
using shape_idx_t = cista::strong<std::uint32_t, struct _shape_idx>;
using shape_offset_t = cista::strong<std::uint32_t, struct _shape_offset>;
using shape_offset_idx_t =
    cista::strong<std::uint32_t, struct _shape_offset_idx>;
using trip_idx_t = cista::strong<std::uint32_t, struct _trip_idx>;
using trip_id_idx_t = cista::strong<std::uint32_t, struct _trip_id_str_idx>;
using transport_idx_t = cista::strong<std::uint32_t, struct _transport_idx>;
using source_idx_t = cista::strong<std::uint16_t, struct _source_idx>;
using day_idx_t = cista::strong<std::uint16_t, struct _day_idx>;
using timezone_idx_t = cista::strong<std::uint16_t, struct _timezone_idx>;
using merged_trips_idx_t =
    cista::strong<std::uint32_t, struct _merged_trips_idx>;
using footpath_idx_t = cista::strong<std::uint32_t, struct _footpath_idx>;
using source_file_idx_t = cista::strong<std::uint32_t, struct _source_file_idx>;

using profile_idx_t = std::uint8_t;
static constexpr auto const kMaxProfiles = profile_idx_t{8};

using rt_trip_idx_t = cista::strong<std::uint32_t, struct _trip_idx>;
using rt_add_trip_id_idx_t =
    cista::strong<std::uint32_t, struct _trip_id_str_idx>;
using rt_route_idx_t = cista::strong<std::uint32_t, struct _rt_route_idx>;
using rt_transport_idx_t =
    cista::strong<std::uint32_t, struct _rt_transport_idx>;
using rt_merged_trips_idx_t =
    cista::strong<std::uint32_t, struct _merged_trips_idx>;

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
using booking_rule_idx_t =
    cista::strong<std::uint32_t, struct _booking_rule_idx>;

using transport_range_t = pair<transport_idx_t, interval<stop_idx_t>>;

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
  CISTA_PRINTABLE(provider, "short_name", "long_name", "url")
  string short_name_, long_name_, url_;
  timezone_idx_t tz_{timezone_idx_t::invalid()};
};

// colors in ARGB layout, 0 thus indicates no color specified
using color_t = cista::strong<std::uint32_t, struct _color>;
struct route_color {
  color_t color_;
  color_t text_color_;
};
inline std::optional<std::string> to_str(color_t const c) {
  return c == 0U ? std::nullopt
                 : std::optional{fmt::format("{:06x}", to_idx(c) & 0x00ffffff)};
}

struct trip_id {
  CISTA_COMPARABLE()
  inline friend std::ostream& operator<<(std::ostream& out, trip_id const tid) {
    return out << "{id=" << tid.id_
               << ", src=" << static_cast<int>(to_idx(tid.src_)) << "}";
  }
  std::string_view id_;
  source_idx_t src_;
};

using match_t = std::vector<geometry_idx_t>;

struct geometry_id {
  CISTA_COMPARABLE()
  CISTA_PRINTABLE(geometry_id, "id", "src")
  string id_;
  source_idx_t src_;
};

struct geometry_trip_idx {
  CISTA_COMPARABLE()
  CISTA_PRINTABLE(geometry_trip_idx, "trip_idx", "geometry_idx")
  trip_idx_t trip_idx_;
  geometry_idx_t geometry_idx_;
};

struct geometry_trip_id {
  CISTA_COMPARABLE()
  CISTA_PRINTABLE(geometry_trip_id, "geometry_id", "trip_id", "src")
  string geometry_id_;
  string trip_id_;
  source_idx_t src_;
};

using geometry_trip_idx_t =
    cista::strong<std::uint32_t, struct _locationtrip_idx_t>;

struct location_id {
  CISTA_COMPARABLE()
  CISTA_PRINTABLE(location_id, "id", "src")
  string id_;
  source_idx_t src_;
};

struct debug {
  inline friend std::ostream& operator<<(std::ostream& out, debug const dbg) {
    return out << dbg.path_ << ":" << dbg.line_from_ << ":" << dbg.line_to_;
  }
  std::string_view path_;
  unsigned line_from_{0U}, line_to_{0U};
};

struct transport {
  CISTA_FRIEND_COMPARABLE(transport)
  CISTA_PRINTABLE(transport, "idx", "day")
  static transport invalid() noexcept { return transport{}; }
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

constexpr u8_minutes operator""_u8_minutes(unsigned long long n) {
  return duration_t{n};
}

constexpr duration_t operator""_minutes(unsigned long long n) {
  return duration_t{n};
}

constexpr duration_t operator""_hours(unsigned long long n) {
  return duration_t{n * 60U};
}

constexpr duration_t operator""_days(unsigned long long n) {
  return duration_t{n * 1440U};
}

using minutes_after_midnight_t = duration_t;

struct stop_window {
  minutes_after_midnight_t start_{0}, end_{0};
};

struct tz_offsets {
  struct season {
    duration_t offset_{0};
    unixtime_t begin_{unixtime_t::min()}, end_{unixtime_t::max()};
    duration_t season_begin_mam_{0};
    duration_t season_end_mam_{0};
  };
  vector<season> seasons_;
  duration_t offset_{0};
};

using timezone = variant<pair<string, void const*>, tz_offsets>;

struct booking_rule {
  CISTA_COMPARABLE()

  uint8_t type_;
  uint16_t prior_notice_duration_min_;
  uint16_t prior_notice_duration_max_;
  uint16_t prior_notice_last_day_;
  duration_t prior_notice_last_time_;
  uint16_t prior_notice_start_day_;
  duration_t prior_notice_start_time_;
  bitfield_idx_t bitfield_idx_;
};

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

enum class location_type : std::uint8_t {
  kGeneratedTrack,  // track generated from track number (i.e. HRD), no separate
                    // coordinate from parent. Has to be connected manually in
                    // routing initialization (only links to parent are given).
  kTrack,  // track from input data (i.e. GTFS) with separate coordinate from
           // parent. No manual connection in routing initialization or
           // additional links between parent<->child necessary.
  kStation
};

enum stop_type { kPickup, kDropoff };

enum pickup_dropoff_type : std::uint8_t {
  kRegularType = 0,
  kUnavailableType = 1,
  kPhoneAgencyType = 2,
  kCoordinateWithDriverType = 3
};

enum class event_type { kArr, kDep };

enum class direction {
  kForward,
  kBackward  // start = final arrival, destination = journey departure
};

inline constexpr direction flip(direction const d) noexcept {
  return d == direction::kForward ? direction::kBackward : direction::kForward;
}

inline std::string_view to_str(direction const d) {
  return d == direction::kForward ? "FWD" : "BWD";
}

template <direction D, typename Collection>
auto to_range(Collection const& c) {
  if constexpr (D == direction::kForward) {
    return it_range{c.begin(), c.end()};
  } else {
    return it_range{c.rbegin(), c.rend()};
  }
}

using transport_mode_id_t = std::int32_t;

using via_offset_t = std::uint8_t;

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
                                nigiri::duration_t const& t) {
  return out << std::chrono::duration_cast<nigiri::i32_minutes>(t);
}

inline std::ostream& operator<<(std::ostream& out,
                                nigiri::unixtime_t const& t) {
  date::to_stream(out, "%F %R", t);
  return out;
}

}  // namespace std::chrono

#include <iostream>

namespace nigiri {

struct delta {
  explicit delta(duration_t const d)
      : days_{static_cast<std::uint16_t>(d.count() / 1440)},
        mam_{static_cast<std::uint16_t>(d.count() % 1440)} {
    assert(d.count() >= 0);
  }

  explicit delta(std::uint16_t const minutes)
      : days_{static_cast<std::uint16_t>(minutes / 1440U)},
        mam_{static_cast<std::uint16_t>(minutes % 1440U)} {}

  delta(std::uint16_t const day, std::uint16_t const mam)
      : days_{day}, mam_{mam} {}

  std::uint16_t value() const {
    return *reinterpret_cast<std::uint16_t const*>(this);
  }

  std::int16_t days() const { return days_; }
  std::int16_t mam() const { return mam_; }

  friend std::ostream& operator<<(std::ostream& out, delta const& d) {
    return out << duration_t{static_cast<duration_t::rep>(d.mam_)} << "."
               << d.days_;
  }

  friend delta operator-(delta const a, delta const b) {
    return delta{static_cast<std::uint16_t>((a.days_ - b.days_) * 1440U +
                                            (a.mam_ - b.mam_))};
  }

  cista::hash_t hash() const {
    return cista::hash_combine(cista::BASE_HASH, value());
  }

  duration_t as_duration() const { return days() * 1_days + mam() * 1_minutes; }

  std::int16_t count() const { return days_ * 1440U + mam_; }

  friend bool operator<(delta const a, delta const b) {
    return a.value() < b.value();
  }
  friend bool operator==(delta const a, delta const b) {
    return a.value() == b.value();
  }

  std::uint16_t days_ : 5;
  std::uint16_t mam_ : 11;
};

template <std::size_t NMaxTypes>
constexpr auto static_type_hash(delta const*,
                                cista::hash_data<NMaxTypes> h) noexcept {
  return h.combine(cista::hash("nigiri::delta"));
}

template <typename Ctx>
inline void serialize(Ctx&, delta const*, cista::offset_t const) {}

template <typename Ctx>
inline void deserialize(Ctx const&, delta*) {}

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
      [t](pair<string, void const*> const& x) {
        return to_local_time_tz(
            reinterpret_cast<date::time_zone const*>(x.second), t);
      }});
}

}  // namespace nigiri

template <>
struct fmt::formatter<nigiri::duration_t> : ostream_formatter {};

template <>
struct fmt::formatter<nigiri::unixtime_t> : ostream_formatter {};

template <>
struct fmt::formatter<nigiri::debug> : ostream_formatter {};

template <>
struct fmt::formatter<nigiri::delta> : ostream_formatter {};

template <>
struct fmt::formatter<nigiri::transport> : ostream_formatter {};