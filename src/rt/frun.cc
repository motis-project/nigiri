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

std::string_view frun::run_stop::name() const noexcept {
  auto const l = get_location_idx();
  auto const type = fr_->tt_->locations_.types_.at(l);
  auto const p =
      (type == location_type::kGeneratedTrack || type == location_type::kTrack)
          ? fr_->tt_->locations_.parents_.at(l)
          : l;
  return fr_->tt_->locations_.names_.at(p).view();
}

std::string_view frun::run_stop::id() const noexcept {
  auto const l = get_location_idx();
  auto const type = fr_->tt_->locations_.types_.at(l);
  return fr_->tt_->locations_.ids_
      .at(type == location_type::kGeneratedTrack
              ? fr_->tt_->locations_.parents_.at(l)
              : l)
      .view();
}

std::string_view frun::run_stop::track() const noexcept {
  auto const l = get_location_idx();
  return (fr_->tt_->locations_.types_.at(l) == location_type::kTrack ||
          fr_->tt_->locations_.types_.at(l) == location_type::kGeneratedTrack)
             ? fr_->tt_->locations_.names_.at(l).view()
             : "";
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

unixtime_t frun::run_stop::time(event_type const ev_type) const noexcept {
  assert(fr_->size() >= stop_idx_);
  return (fr_->is_rt() && fr_->rtt_ != nullptr)
             ? fr_->rtt_->unix_event_time(fr_->rt_, stop_idx_, ev_type)
             : fr_->tt_->event_time(fr_->t_, stop_idx_, ev_type);
}

std::string_view frun::run_stop::line() const noexcept {
  if (fr_->is_rt() && fr_->rtt_ != nullptr) {
    auto const rt_line = fr_->rtt_->rt_transport_line_.at(fr_->rt_);
    return rt_line.empty() ? scheduled_line() : rt_line.view();
  } else {
    return scheduled_line();
  }
}

std::string_view frun::run_stop::scheduled_line() const noexcept {
  if (!fr_->is_scheduled()) {
    return "";
  }

  auto const section_lines =
      fr_->tt_->transport_section_lines_.at(fr_->t_.t_idx_);
  if (section_lines.empty()) {
    return "";
  } else {
    auto const line_idx = section_lines.size() == 1U
                              ? section_lines[0]
                              : section_lines.at(stop_idx_);
    return fr_->tt_->trip_lines_.at(line_idx).view();
  }
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

  // Print stop index, location name.
  fmt::print(out, "  {:2}: {:7} {:.<48}", stp.stop_idx_, stp.get_location().id_,
             stp.name());

  // Print arrival (or whitespace if there's none).
  if (stp.stop_idx_ != 0U) {
    auto const scheduled = stp.scheduled_time(event_type::kArr);
    auto const rt = stp.time(event_type::kArr);
    fmt::print(out, "{}a: {} [{}]", (stp.out_allowed() ? ' ' : '-'),
               date::format("%d.%m %R", scheduled),
               date::format("%d.%m %R", to_local_time(tz, scheduled)));
    if (stp.fr_->is_rt() && stp.fr_->rtt_ != nullptr) {  // RT if available.
      fmt::print(out, "  RT {} [{}]", date::format("%d.%m %R", rt),
                 date::format("%d.%m %R", to_local_time(tz, rt)));
    }
  } else if (stp.fr_->is_rt() && stp.fr_->rtt_ != nullptr) {
    // Skipped w/ RT info.
    fmt::print(out, "                            ");
    fmt::print(out, "                               ");
  } else {
    // Skipped w/o RT info.
    fmt::print(out, "                             ");
  }

  // Print departure (or whitespace if there's none).
  if (stp.stop_idx_ != stp.fr_->size() - 1U) {
    fmt::print(out, " ");
    auto const scheduled = stp.scheduled_time(event_type::kDep);
    auto const rt = stp.time(event_type::kDep);
    fmt::print(out, "{}d: {} [{}]", (stp.in_allowed() ? ' ' : '-'),
               date::format("%d.%m %R", scheduled),
               date::format("%d.%m %R", to_local_time(tz, scheduled)));
    if (stp.fr_->is_rt() && stp.fr_->rtt_ != nullptr) {  // RT if available.
      fmt::print(out, "  RT {} [{}]", date::format("%d.%m %R", rt),
                 date::format("%d.%m %R", to_local_time(tz, rt)));
    }
  }

  // Print trip info.
  if (stp.fr_->is_scheduled() && stp.stop_idx_ != stp.fr_->size() - 1U) {
    auto const& tt = *stp.fr_->tt_;
    auto const& trip_section =
        tt.transport_to_trip_section_.at(stp.fr_->t_.t_idx_);
    auto const& merged_trips = tt.merged_trips_.at(
        trip_section.size() == 1U ? trip_section[0]
                                  : trip_section.at(stp.stop_idx_));

    out << "  [";
    for (auto const& trip_idx : merged_trips) {
      auto j = 0U;

      for (auto const [dbg, id] :
           utl::zip(tt.trip_debug_.at(trip_idx), tt.trip_ids_.at(trip_idx))) {
        if (j++ != 0) {
          out << ", ";
        }
        out << "{name=" << tt.trip_display_names_.at(trip_idx).view()
            << ", day=";
        date::to_stream(out, "%F",
                        tt.internal_interval_days().from_ +
                            to_idx(stp.fr_->t_.day_) * 1_days);
        out << ", id=" << tt.trip_id_strings_.at(id).view()
            << ", src=" << static_cast<int>(to_idx(tt.trip_id_src_.at(id)));
        out << "}";
      }
    }
    out << "]";
  }

  return out;
}

std::ostream& operator<<(std::ostream& out, frun const& fr) {
  for (auto const stp : fr) {
    out << stp << "\n";
  }
  return out;
}

}  // namespace nigiri::rt