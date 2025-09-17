#pragma once

#include <functional>
#include <iosfwd>

#include "geo/latlng.h"

#include "nigiri/common/interval.h"
#include "nigiri/location.h"
#include "nigiri/rt/run.h"
#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri {
struct shapes_storage;
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

struct frun;

struct run_stop {
  stop get_stop() const;
  stop get_scheduled_stop() const;
  location get_location() const;
  geo::latlng pos() const;
  location_idx_t get_location_idx() const;
  location_idx_t get_scheduled_location_idx() const;
  std::string_view name() const;
  std::string_view track() const;
  std::string_view id() const;
  std::pair<date::sys_days, duration_t> get_trip_start(
      event_type = event_type::kDep) const;

  provider_idx_t get_provider_idx(event_type = event_type::kDep) const;
  provider const& get_provider(event_type = event_type::kDep) const;
  trip_idx_t get_trip_idx(event_type = event_type::kDep) const;
  route_id_idx_t get_route_id(event_type = event_type::kDep) const;
  std::optional<route_type_t> route_type(event_type = event_type::kDep) const;
  std::string_view route_short_name(event_type = event_type::kDep) const;
  std::string_view route_long_name(event_type = event_type::kDep) const;
  std::string_view trip_short_name(event_type = event_type::kDep) const;
  std::string_view display_name(event_type = event_type::kDep) const;
  run_stop get_last_trip_stop(event_type = event_type::kDep) const;

  unixtime_t scheduled_time(event_type) const;
  unixtime_t time(event_type) const;
  duration_t delay(event_type) const;
  timezone_idx_t get_tz(event_type) const;

  std::string_view line(event_type = event_type::kDep) const;
  std::string_view scheduled_line(event_type = event_type::kDep) const;
  std::string_view direction(event_type = event_type::kDep) const;

  clasz get_clasz(event_type = event_type::kDep) const;
  clasz get_scheduled_clasz(event_type = event_type::kDep) const;

  bool bikes_allowed(event_type = event_type::kDep) const;
  bool cars_allowed(event_type = event_type::kDep) const;

  route_color get_route_color(event_type = event_type::kDep) const;

  bool in_allowed() const;
  bool out_allowed() const;
  bool in_allowed_wheelchair() const;
  bool out_allowed_wheelchair() const;
  bool is_cancelled() const;

  bool in_allowed(bool const is_wheelchair) const;
  bool out_allowed(bool const is_wheelchair) const;

  template <enum direction SearchDir>
  bool can_start(bool const is_wheelchair) const {
    if constexpr (SearchDir == direction::kForward) {
      return is_wheelchair ? in_allowed_wheelchair() : in_allowed();
    } else {
      return is_wheelchair ? out_allowed_wheelchair() : out_allowed();
    }
  }

  template <enum direction SearchDir>
  bool can_finish(bool const is_wheelchair) const {
    if constexpr (SearchDir == direction::kForward) {
      return is_wheelchair ? out_allowed_wheelchair() : out_allowed();
    } else {
      return is_wheelchair ? in_allowed_wheelchair() : in_allowed();
    }
  }

  stop_idx_t section_idx(event_type) const;

  timetable const& tt() const;
  rt_timetable const* rtt() const;

  void print(std::ostream&, bool first = false, bool last = false) const;
  bool operator==(run_stop const&) const = default;
  friend std::ostream& operator<<(std::ostream&, run_stop const&);

  frun const* fr_{nullptr};
  stop_idx_t stop_idx_{0U};
};

// Full run. Same as `run` data structure, extended with timetable and
// rt_timetable to be able to look up additional info.
struct frun : public run {
  struct iterator {
    using difference_type = stop_idx_t;
    using value_type = run_stop;
    using pointer = run_stop;
    using reference = run_stop;
    using iterator_category = std::bidirectional_iterator_tag;

    iterator& operator++();
    iterator operator++(int);

    iterator& operator--();
    iterator operator--(int);

    bool operator==(iterator const) const;
    bool operator!=(iterator) const;

    run_stop operator*() const;

    run_stop rs_;
  };
  using const_iterator = iterator;

  std::string_view name() const;
  debug dbg() const;

  frun(timetable const&, rt_timetable const*, run);

  iterator begin() const;
  iterator end() const;

  friend iterator begin(frun const&);
  friend iterator end(frun const&);

  std::reverse_iterator<iterator> rbegin() const;
  std::reverse_iterator<iterator> rend() const;

  friend std::reverse_iterator<iterator> rbegin(frun const&);
  friend std::reverse_iterator<iterator> rend(frun const&);

  stop_idx_t first_valid(stop_idx_t from = 0U) const;
  stop_idx_t last_valid() const;

  stop_idx_t size() const;

  run_stop operator[](stop_idx_t) const;

  trip_id id() const;
  trip_idx_t trip_idx() const;
  clasz get_clasz() const;
  bool is_cancelled() const;

  void for_each_trip(
      std::function<void(trip_idx_t const, interval<stop_idx_t> const)> const&)
      const;
  void for_each_shape_point(
      shapes_storage const*,
      interval<stop_idx_t> const,
      std::function<void(geo::latlng const&)> const&) const;

  void print(std::ostream&, interval<stop_idx_t>);
  friend std::ostream& operator<<(std::ostream&, frun const&);

  static frun from_rt(timetable const&,
                      rt_timetable const*,
                      rt_transport_idx_t const);

  static frun from_t(timetable const&, rt_timetable const*, transport const);

  timetable const* tt_;
  rt_timetable const* rtt_;
};

}  // namespace nigiri::rt
