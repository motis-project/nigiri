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

struct block {
  std::vector<std::pair<std::vector<trip*>, bitfield>> rule_services();
  std::vector<trip*> trips_;
};

using block_map = hash_map<std::string, std::unique_ptr<block>>;

struct frequency {
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

struct trip {
  trip(route const*,
       bitfield const*,
       block*,
       std::string id,
       std::string headsign,
       std::string short_name);

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

  std::vector<std::uint16_t> seq_numbers_;
  std::basic_string<timetable::stop::value_type> stop_seq_;
  std::vector<stop_events> event_times_;
  std::vector<std::string> stop_headsigns_;

  std::optional<std::vector<frequency>> frequency_;
  bool requires_interpolation_{false};
  bool requires_sorting_{false};
  std::uint32_t from_line_{0U}, to_line_{0U};
};

using trip_map = hash_map<std::string, std::unique_ptr<trip>>;

std::pair<trip_map, block_map> read_trips(route_map_t const&,
                                          traffic_days const&,
                                          std::string_view file_content);

void read_frequencies(trip_map&, std::string_view);

}  // namespace nigiri::loader::gtfs
