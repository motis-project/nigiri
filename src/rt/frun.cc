#include "nigiri/rt/frun.h"

#include "nigiri/lookup/get_transport_stop_tz.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::rt {

stop frun::run_stop::get_stop() const noexcept {
  return stop{
      (fr_->is_rt() && fr_->rtt_ != nullptr)
          ? fr_->rtt_->rt_transport_location_seq_[fr_->rt_][stop_idx_]
          : fr_->tt_->route_location_seq_
                [fr_->tt_->transport_route_[fr_->t_.t_idx_]][stop_idx_]};
}

location frun::run_stop::get_location() const noexcept {
  assert(fr_->size() >= stop_idx_);
  return location{*fr_->tt_, get_stop().location_idx()};
}

location_idx_t frun::run_stop::get_location_idx() const noexcept {
  assert(fr_->size() >= stop_idx_);
  return get_stop().location_idx();
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
  return (fr_->is_rt() && fr_->rtt_ != nullptr)
             ? fr_->rtt_->unix_event_time(fr_->rt_, stop_idx_, ev_type)
             : fr_->tt_->event_time(fr_->t_, stop_idx_, ev_type);
}

bool frun::run_stop::in_allowed() const noexcept {
  return get_stop().in_allowed();
}

bool frun::run_stop::out_allowed() const noexcept {
  return get_stop().out_allowed();
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

std::string_view frun::name() const noexcept {
  return (is_rt() && rtt_ != nullptr) ? rtt_->transport_name(*tt_, rt_)
                                      : tt_->transport_name(t_.t_idx_);
}

debug frun::dbg() const noexcept {
  return (is_rt() && rtt_ != nullptr) ? rtt_->dbg(*tt_, rt_)
                                      : tt_->dbg(t_.t_idx_);
}

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
      (is_rt() && rtt_ != nullptr)
          ? rtt_->rt_transport_location_seq_[rt_].size()
          : tt_->route_location_seq_[tt_->transport_route_[t_.t_idx_]].size());
}

frun::run_stop frun::operator[](stop_idx_t const i) const noexcept {
  return run_stop{this, i};
}

std::ostream& operator<<(std::ostream& out, frun::run_stop const& stp) {
  auto const& tz = stp.fr_->tt_->locations_.timezones_.at(get_transport_stop_tz(
      *stp.fr_->tt_, stp.fr_->t_.t_idx_, stp.get_location().l_));
  fmt::print(out, "{:2}: {:7} {:.<48}  ", stp.stop_idx_, stp.get_location().id_,
             stp.get_location().name_);
  if (stp.stop_idx_ != 0U) {
    auto const st = stp.scheduled_time(event_type::kArr);
    auto const rt = stp.real_time(event_type::kArr);
    fmt::print(out, "{}a: {} [{}]  RT{} {} [{}]",
               (stp.out_allowed() ? ' ' : '-'), date::format("%d.%m %R", st),
               date::format("%d.%m %R", to_local_time(tz, st)),
               (stp.fr_->is_rt() && stp.fr_->rtt_ != nullptr) ? '*' : '_',
               date::format("%d.%m %R", rt),
               date::format("%d.%m %R", to_local_time(tz, rt)));
  } else {
    fmt::print(out,
               "                                                            ");
  }
  if (stp.stop_idx_ != stp.fr_->size() - 1U) {
    fmt::print(out, "      ");
    auto const st = stp.scheduled_time(event_type::kDep);
    auto const rt = stp.real_time(event_type::kDep);
    fmt::print(out, "{}d: {} [{}]  RT{} {} [{}]",
               (stp.in_allowed() ? ' ' : '-'), date::format("%d.%m %R", st),
               date::format("%d.%m %R", to_local_time(tz, st)),
               (stp.fr_->is_rt() && stp.fr_->rtt_ != nullptr) ? '*' : '_',
               date::format("%d.%m %R", rt),
               date::format("%d.%m %R", to_local_time(tz, rt)));
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, frun const& fr) {
  for (auto const& stp : fr) {
    out << stp << "\n";
  }
  return out;
}

}  // namespace nigiri::rt