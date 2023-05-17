#pragma once

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "cista/reflection/comparable.h"

#include "nigiri/loader/gtfs/flat_map.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/services.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

struct trip;

using gtfs_trip_idx_t = cista::strong<std::uint32_t, struct _gtfs_trip_idx>;

struct trip_data;

struct block {
  std::vector<std::pair<std::basic_string<gtfs_trip_idx_t>, bitfield>>
  rule_services(trip_data&);

  std::vector<gtfs_trip_idx_t> trips_;
};

using stop_seq_t = std::basic_string<timetable::stop::value_type>;

struct frequency {
  unsigned number_of_iterations() const {
    return static_cast<unsigned>((end_time_ - start_time_) / headway_);
  }
  minutes_after_midnight_t get_iteration_start_time(
      unsigned const iteration) const {
    return start_time_ + iteration * headway_;
  }
  friend bool operator==(frequency const&, frequency const&) = default;
  minutes_after_midnight_t start_time_{0U};
  minutes_after_midnight_t end_time_{0U};
  duration_t headway_{0U};
  enum class schedule_relationship : std::uint8_t {
    kScheduled,
    kUnscheduled
  } schedule_relationship_;
};

struct stop_events {
  minutes_after_midnight_t arr_{kInterpolate}, dep_{kInterpolate};
};

struct trip_seq_data {
  vecvec<gtfs_trip_idx_t, timetable::stop::value_type> stops_seq_;
  vecvec<gtfs_trip_idx_t, std::uint16_t> seq_numbers_;
  vecvec<gtfs_trip_idx_t, stop_events> stop_events_;
  vecvec<gtfs_trip_idx_t, trip_direction_idx_t> stop_headsigns_;
};

struct trip {
  trip(route const*,
       bitfield const*,
       block*,
       std::string id,
       std::string headsign,
       std::string short_name);

  trip(trip&&) = default;
  trip& operator=(trip&&) = default;

  trip(trip const&) = delete;
  trip& operator=(trip const&) = delete;

  ~trip() = default;

  void interpolate();

  void print_stop_times(std::ostream&,
                        timetable const&,
                        unsigned indent = 0) const;

  std::string display_name(timetable const&) const;

  route const* route_{nullptr};
  bitfield const* service_{nullptr};
  block* block_{nullptr};
  std::string id_;
  std::string headsign_;
  std::string short_name_;
  std::optional<std::vector<frequency>> frequency_;
  bool requires_interpolation_{false};
  bool requires_sorting_{false};
  std::uint32_t from_line_{0U}, to_line_{0U};
};

struct trip_data {
  trip const& get(gtfs_trip_idx_t const idx) const { return data_[idx]; }
  trip& get(gtfs_trip_idx_t const idx) { return data_[idx]; }
  trip const& get(std::string_view id) const { return data_[trips_.at(id)]; }
  trip& get(std::string_view id) { return data_[trips_.at(id)]; }
  hash_map<std::string, gtfs_trip_idx_t> trips_;
  hash_map<std::string, std::unique_ptr<block>> blocks_;
  vector_map<gtfs_trip_idx_t, trip> data_;
};

trip_data read_trips(route_map_t const&,
                     traffic_days const&,
                     std::string_view file_content);

void read_frequencies(trip_data&, std::string_view);

}  // namespace nigiri::loader::gtfs
