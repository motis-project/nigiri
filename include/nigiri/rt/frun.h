#pragma once

#include <iosfwd>
#include <span>

#include "geo/latlng.h"

#include "nigiri/common/interval.h"
#include "nigiri/location.h"
#include "nigiri/rt/run.h"
#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

// Full run. Same as `run` data structure, extended with timetable and
// rt_timetable to be able to look up additional info.
struct frun : public run {
  struct run_stop {
    stop get_stop() const noexcept;
    location get_location() const noexcept;
    geo::latlng pos() const noexcept;
    location_idx_t get_location_idx() const noexcept;
    std::string_view name() const noexcept;
    std::string_view track() const noexcept;
    std::string_view id() const noexcept;

    provider const& get_provider(event_type = event_type::kDep) const noexcept;
    trip_idx_t get_trip_idx(event_type = event_type::kDep) const noexcept;
    std::string_view trip_display_name(
        event_type = event_type::kDep) const noexcept;

    unixtime_t scheduled_time(event_type) const noexcept;
    unixtime_t time(event_type) const noexcept;
    duration_t delay(event_type) const noexcept;

    std::string_view line(event_type = event_type::kDep) const noexcept;
    std::string_view scheduled_line(
        event_type = event_type::kDep) const noexcept;
    std::string_view direction(event_type = event_type::kDep) const noexcept;

    clasz get_clasz(event_type = event_type::kDep) const noexcept;
    clasz get_scheduled_clasz(event_type = event_type::kDep) const noexcept;

    bool bikes_allowed(event_type = event_type::kDep) const noexcept;

    route_color get_route_color(event_type = event_type::kDep) const noexcept;

    bool in_allowed() const noexcept;
    bool out_allowed() const noexcept;
    bool in_allowed_wheelchair() const noexcept;
    bool out_allowed_wheelchair() const noexcept;
    bool is_canceled() const noexcept;

    bool in_allowed(bool const is_wheelchair) const noexcept;
    bool out_allowed(bool const is_wheelchair) const noexcept;

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

    stop_idx_t section_idx(event_type) const noexcept;

    timetable const& tt() const noexcept;
    rt_timetable const* rtt() const noexcept;

    void print(std::ostream&, bool first = false, bool last = false) const;
    bool operator==(run_stop const&) const = default;
    friend std::ostream& operator<<(std::ostream&, run_stop const&);

    frun const* fr_{nullptr};
    stop_idx_t stop_idx_{0U};
  };

  struct iterator {
    using difference_type = stop_idx_t;
    using value_type = run_stop;
    using pointer = run_stop;
    using reference = run_stop;
    using iterator_category = std::forward_iterator_tag;

    iterator& operator++() noexcept;
    iterator operator++(int) noexcept;

    bool operator==(iterator const o) const noexcept;
    bool operator!=(iterator o) const noexcept;

    run_stop operator*() const noexcept;

    run_stop rs_;
  };
  using const_iterator = iterator;

  std::string_view name() const noexcept;
  debug dbg() const noexcept;

  frun(timetable const&, rt_timetable const*, run);

  iterator begin() const noexcept;
  iterator end() const noexcept;

  friend iterator begin(frun const& fr) noexcept;
  friend iterator end(frun const& fr) noexcept;

  stop_idx_t first_valid(stop_idx_t from = 0U) const;
  stop_idx_t last_valid() const;

  stop_idx_t size() const noexcept;

  run_stop operator[](stop_idx_t) const noexcept;

  trip_id id() const noexcept;
  trip_idx_t trip_idx() const;
  clasz get_clasz() const noexcept;

  std::span<geo::latlng const> get_shape(shapes_storage_t const&, trip_idx_t, interval<stop_idx_t> const&) const;

  void print(std::ostream&, interval<stop_idx_t> stop_range);
  friend std::ostream& operator<<(std::ostream&, frun const&);

  static frun from_rt(timetable const&,
                      rt_timetable const*,
                      rt_transport_idx_t const) noexcept;

  static frun from_t(timetable const&,
                     rt_timetable const*,
                     transport const) noexcept;

  timetable const* tt_;
  rt_timetable const* rtt_;
  mutable std::array<geo::latlng, 2> shape_cache_;  // FIXME Remove cache from class (maybe part of 'shape'?)
};

}  // namespace nigiri::rt
