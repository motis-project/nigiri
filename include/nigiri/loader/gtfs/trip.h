#pragma once

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "utl/enumerate.h"

#include "cista/reflection/comparable.h"

#include "nigiri/loader/gtfs/flat_map.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/services.h"
#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/translations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct trip;

using gtfs_trip_idx_t = cista::strong<std::uint32_t, struct _gtfs_trip_idx>;

struct trip_data;

static auto const kSingleTripBikesAllowed = bitvec{"1"};
static auto const kSingleTripBikesNotAllowed = bitvec{"0"};

struct block {
  std::vector<std::pair<basic_string<gtfs_trip_idx_t>, bitfield>> rule_services(
      trip_data&);

  std::vector<gtfs_trip_idx_t> trips_;
};

using stop_seq_t = basic_string<stop::value_type>;

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

struct stop_time_window {
  booking_rule_idx_t pickup_booking_rule_{booking_rule_idx_t::invalid()};
  booking_rule_idx_t drop_off_booking_rule_{booking_rule_idx_t::invalid()};
  minutes_after_midnight_t start_, end_;
};

struct trip {
  trip(route_id_idx_t,
       bitfield const*,
       block*,
       std::string id,
       translation_idx_t headsign,
       translation_idx_t short_name,
       direction_id_t,
       shape_idx_t,
       bool bikes_allowed,
       bool cars_allowed);

  trip(trip&&) = default;
  trip& operator=(trip&&) = default;

  trip(trip const&) = delete;
  trip& operator=(trip const&) = delete;

  ~trip() = default;

  void print_stop_times(std::ostream&,
                        timetable const&,
                        unsigned indent = 0) const;

  std::string display_name() const;

  bool has_seated_transfers() const;

  route_id_idx_t route_{route_id_idx_t::invalid()};
  bitfield const* service_{nullptr};
  block* block_{nullptr};
  std::string id_;
  translation_idx_t headsign_;
  direction_id_t direction_id_{direction_id_t::invalid()};
  translation_idx_t short_name_;
  shape_idx_t shape_idx_;

  stop_seq_t stop_seq_;
  std::vector<std::uint16_t> seq_numbers_;
  std::vector<stop_events> event_times_;
  std::vector<translation_idx_t> stop_headsigns_;
  std::vector<double> distance_traveled_;

  std::vector<flex_stop_t> flex_stops_;
  std::vector<stop_time_window> flex_time_windows_;

  std::vector<gtfs_trip_idx_t> seated_out_, seated_in_;

  std::optional<std::vector<frequency>> frequency_;
  bool requires_interpolation_{false};
  bool requires_sorting_{false};
  bool bikes_allowed_{false};
  bool cars_allowed_{false};
  std::uint32_t from_line_{0U}, to_line_{0U};

  trip_idx_t trip_idx_{trip_idx_t::invalid()};
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

enum class interpolate_result { kOk, kErrorLastMissing, kErrorFirstMissing };

interpolate_result interpolate(std::vector<stop_events>&);

trip_data read_trips(source_idx_t,
                     source_file_idx_t,
                     timetable&,
                     translator&,
                     route_map_t const&,
                     traffic_days_t const&,
                     shape_loader_state const&,
                     std::string_view file_content,
                     std::array<bool, kNumClasses> const& bikes_allowed_default,
                     std::array<bool, kNumClasses> const& cars_allowed_default,
                     script_runner const& = script_runner{});

void read_frequencies(trip_data&, std::string_view);

}  // namespace nigiri::loader::gtfs
