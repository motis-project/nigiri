#pragma once

#include "nigiri/location.h"
#include "nigiri/rt/run.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

// Full run. Same as `run` data structure, extended with timetable and
// rt_timetable to be able to look up additional info.
struct frun : public run {
  struct run_stop {
    location get_location() const noexcept;
    unixtime_t scheduled_time(event_type const ev_type) const noexcept;
    unixtime_t real_time(event_type const ev_type) const noexcept;
    bool operator==(run_stop const&) const = default;

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

  frun(timetable const&, rt_timetable const*, run);
  frun(timetable const&, rt_timetable const&, rt_transport_idx_t const);
  frun(timetable const&, rt_timetable const*, transport const);

  iterator begin() const noexcept;
  iterator end() const noexcept;

  friend iterator begin(frun const& fr) noexcept;
  friend iterator end(frun const& fr) noexcept;

  stop_idx_t size() const noexcept;

  timetable const* tt_;
  rt_timetable const* rtt_;
};

}  // namespace nigiri::rt