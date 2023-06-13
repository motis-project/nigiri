#include "nigiri/rt/frun.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::rt {

location frun::run_stop::get_location() const noexcept {
  assert(fr_->size() >= stop_idx_);
  return location{
      *fr_->tt_,
      stop{fr_->is_rt()
               ? fr_->rtt_->rt_transport_location_seq_[fr_->rt_][stop_idx_]
               : fr_->tt_->route_location_seq_
                     [fr_->tt_->transport_route_[fr_->t_.t_idx_]][stop_idx_]}
          .location_idx()};
}

unixtime_t frun::run_stop::scheduled_time(
    event_type const ev_type) const noexcept {
  assert(fr_->size() >= stop_idx_);
  return fr_->is_scheduled()
             ? fr_->tt_->event_time(fr_->t_, stop_idx_, ev_type)
             : fr_->rtt_->unix_event_time(fr_->rt_, stop_idx_, ev_type);
}

unixtime_t frun::run_stop::real_time(event_type const ev_type) const noexcept {
  assert(fr_->size() >= stop_idx_);
  return fr_->is_rt() ? fr_->rtt_->unix_event_time(fr_->rt_, stop_idx_, ev_type)
                      : fr_->tt_->event_time(fr_->t_, stop_idx_, ev_type);
}

frun::iterator& frun::iterator::operator++() noexcept {
  ++rs_.stop_idx_;
  return *this;
}

frun::iterator frun::iterator::operator++(int) noexcept {
  auto r = *this;
  ++(*this);
  return r;
}

bool frun::iterator::operator==(iterator const o) const noexcept {
  return rs_ == o.rs_;
}

bool frun::iterator::operator!=(iterator o) const noexcept {
  return !(*this == o);
}

frun::run_stop frun::iterator::operator*() const noexcept { return rs_; }

frun::frun(timetable const& tt, rt_timetable const* rtt, run r)
    : run{r}, tt_{&tt}, rtt_{rtt} {}

frun::frun(timetable const& tt,
           rt_timetable const& rtt,
           rt_transport_idx_t const rt_t)
    : run{.t_ = rtt.resolve_static(rt_t), .rt_ = rt_t}, tt_{&tt}, rtt_{&rtt} {}

frun::frun(timetable const& tt, rt_timetable const* rtt, transport const t)
    : run{.t_ = t,
          .rt_ = rtt == nullptr ? rt_transport_idx_t::invalid()
                                : rtt->resolve_rt(t)},
      tt_{&tt},
      rtt_{rtt} {}

frun::iterator frun::begin() const noexcept {
  return iterator{run_stop{.fr_ = this, .stop_idx_ = 0U}};
}

frun::iterator frun::end() const noexcept {
  return iterator{run_stop{.fr_ = this, .stop_idx_ = size()}};
}

frun::iterator begin(frun const& fr) noexcept { return fr.begin(); }
frun::iterator end(frun const& fr) noexcept { return fr.end(); }

stop_idx_t frun::size() const noexcept {
  return static_cast<stop_idx_t>(
      is_rt()
          ? rtt_->rt_transport_location_seq_[rt_].size()
          : tt_->route_location_seq_[tt_->transport_route_[t_.t_idx_]].size());
}

}  // namespace nigiri::rt