#pragma once

#include <functional>
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

struct stop_time {
  stop_time();
  stop_time(location_idx_t,
            std::string headsign,
            minutes_after_midnight_t arr_time,
            bool out_allowed,
            minutes_after_midnight_t dep_time,
            bool in_allowed);

  struct ev {
    minutes_after_midnight_t time_{kInterpolate};
    bool in_out_allowed_{false};
  };

  location_idx_t stop_{location_idx_t::invalid()};
  std::string headsign_;
  ev arr_, dep_;
};

struct frequency {
  minutes_after_midnight_t start_time_{0U};
  minutes_after_midnight_t end_time_{0U};
  duration_t headway_{0U};
  enum class schedule_relationship : std::uint8_t {
    kScheduled,
    kUnscheduled
  } schedule_relationship_;
};

struct trip {
  using stop_seq = std::basic_string<timetable::stop::value_type>;
  using stop_seq_numbers = std::vector<unsigned>;

  trip(route const*,
       bitfield const*,
       block*,
       std::string id,
       std::string headsign,
       std::string short_name,
       std::uint32_t line);

  void interpolate();

  stop_seq stops() const;
  stop_seq_numbers seq_numbers() const;

  void expand_frequencies(
      std::function<void(trip const&, frequency::schedule_relationship)> const&)
      const;

  void print_stop_times(std::ostream&,
                        timetable const&,
                        unsigned indent = 0) const;

  route const* route_;
  bitfield const* service_;
  block* block_;
  std::string id_;
  std::string headsign_;
  std::string short_name_;
  flat_map<stop_time> stop_times_;
  std::uint32_t line_;
  std::optional<std::vector<frequency>> frequency_;
};

using trip_map = hash_map<std::string, std::unique_ptr<trip>>;

std::pair<trip_map, block_map> read_trips(route_map_t const&,
                                          traffic_days const&,
                                          std::string_view file_content);

void read_frequencies(trip_map&, std::string_view);

}  // namespace nigiri::loader::gtfs
