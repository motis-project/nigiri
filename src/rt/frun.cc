#include "nigiri/rt/frun.h"

#include <iterator>
#include <span>
#include <variant>

#include "utl/overloaded.h"
#include "utl/verify.h"

#include "nigiri/lookup/get_transport_stop_tz.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"

namespace nigiri::rt {

stop run_stop::get_stop() const {
  assert(fr_->size() > stop_idx_);
  return stop{
      (fr_->is_rt() && rtt() != nullptr)
          ? rtt()->rt_transport_location_seq_[fr_->rt_][stop_idx_]
          : tt().route_location_seq_[tt().transport_route_[fr_->t_.t_idx_]]
                                    [stop_idx_]};
}

stop run_stop::get_scheduled_stop() const {
  assert(fr_->size() > stop_idx_);
  return fr_->is_scheduled()
             ? tt().route_location_seq_[tt().transport_route_[fr_->t_.t_idx_]]
                                       [stop_idx_]
             : rtt()->rt_transport_location_seq_[fr_->rt_][stop_idx_];
}

std::string_view run_stop::name() const {
  auto const l = get_location_idx();
  auto const type = tt().locations_.types_.at(l);
  auto const p =
      (type == location_type::kGeneratedTrack || type == location_type::kTrack)
          ? tt().locations_.parents_.at(l)
          : l;
  return tt().locations_.names_.at(p).view();
}

std::string_view run_stop::id() const {
  auto const l = get_location_idx();
  auto const type = tt().locations_.types_.at(l);
  return tt()
      .locations_.ids_
      .at(type == location_type::kGeneratedTrack
              ? tt().locations_.parents_.at(l)
              : l)
      .view();
}

std::string_view run_stop::track() const {
  auto const l = get_location_idx();
  return (tt().locations_.types_.at(l) == location_type::kTrack ||
          tt().locations_.types_.at(l) == location_type::kGeneratedTrack)
             ? tt().locations_.names_.at(l).view()
             : "";
}

location run_stop::get_location() const {
  assert(fr_->size() > stop_idx_);
  return location{*fr_->tt_, get_stop().location_idx()};
}

geo::latlng run_stop::pos() const {
  assert(fr_->size() > stop_idx_);
  return fr_->tt_->locations_.coordinates_[get_stop().location_idx()];
}

location_idx_t run_stop::get_location_idx() const {
  assert(fr_->size() > stop_idx_);
  return get_stop().location_idx();
}

location_idx_t run_stop::get_scheduled_location_idx() const {
  assert(fr_->size() > stop_idx_);
  return get_scheduled_stop().location_idx();
}

unixtime_t run_stop::scheduled_time(event_type const ev_type) const {
  assert(fr_->size() > stop_idx_);
  return fr_->is_scheduled()
             ? tt().event_time(fr_->t_, stop_idx_, ev_type)
             : rtt()->unix_event_time(fr_->rt_, stop_idx_, ev_type);
}

unixtime_t run_stop::time(event_type const ev_type) const {
  assert(fr_->size() > stop_idx_);
  return (fr_->is_rt() && rtt() != nullptr)
             ? rtt()->unix_event_time(fr_->rt_, stop_idx_, ev_type)
             : tt().event_time(fr_->t_, stop_idx_, ev_type);
}

duration_t run_stop::delay(event_type const ev_type) const {
  assert(fr_->size() > stop_idx_);
  return time(ev_type) - scheduled_time(ev_type);
}

stop_idx_t run_stop::section_idx(event_type const ev_type) const {
  return static_cast<stop_idx_t>(ev_type == event_type::kDep ? stop_idx_
                                                             : stop_idx_ - 1);
}

std::string_view run_stop::line() const {
  if (fr_->is_rt() && rtt() != nullptr) {
    auto const rt_line = rtt()->rt_transport_line_.at(fr_->rt_);
    return rt_line.empty() ? scheduled_line() : rt_line.view();
  } else {
    return scheduled_line();
  }
}

std::string_view run_stop::direction() const {
  if (fr_->is_scheduled()) {
    auto const direction_sections =
        tt().transport_section_directions_.at(fr_->t_.t_idx_);
    auto direction = std::string_view{};
    if (!direction_sections.empty()) {
      auto const direction_idx =
          direction_sections.size() == 1U
              ? direction_sections.at(0)
              : direction_sections.at(section_idx(ev_type));
      if (direction_idx != trip_direction_idx_t::invalid()) {
        direction = tt().trip_directions_.at(direction_idx)
                        .apply(utl::overloaded{
                            [&](trip_direction_string_idx_t const i) {
                              return tt().trip_direction_strings_.at(i).view();
                            },
                            [&](location_idx_t const i) {
                              return tt().locations_.names_.at(i).view();
                            }});
      }
    }
    if (!direction.empty()) {
      return direction;
    }
  }
  if (rtt() != nullptr || fr_->is_scheduled()) {
    return run_stop{.fr_ = fr_,
                    .stop_idx_ = static_cast<stop_idx_t>(fr_->size() - 1U)}
        .name();
  }
  return "";
}

bool run_stop::is_cancelled() const { return get_stop().is_cancelled(); }

bool run_stop::in_allowed() const { return get_stop().in_allowed(); }

bool run_stop::out_allowed() const { return get_stop().out_allowed(); }

bool run_stop::in_allowed_wheelchair() const {
  return get_stop().in_allowed_wheelchair();
}

bool run_stop::out_allowed_wheelchair() const {
  return get_stop().out_allowed_wheelchair();
}

bool run_stop::in_allowed(bool const wheelchair) const {
  return wheelchair ? in_allowed_wheelchair() : in_allowed();
}

bool run_stop::out_allowed(bool const wheelchair) const {
  return wheelchair ? out_allowed_wheelchair() : out_allowed();
}

timetable const& run_stop::tt() const { return *fr_->tt_; }
rt_timetable const* run_stop::rtt() const { return fr_->rtt_; }

frun::iterator& frun::iterator::operator++() {
  do {
    ++rs_.stop_idx_;
  } while (rs_.stop_idx_ != rs_.fr_->stop_range_.to_ && rs_.is_cancelled());
  return *this;
}

frun::iterator frun::iterator::operator++(int) {
  auto r = *this;
  ++(*this);
  return r;
}

frun::iterator& frun::iterator::operator--() {
  do {
    --rs_.stop_idx_;
  } while (rs_.stop_idx_ !=
               static_cast<stop_idx_t>(rs_.fr_->stop_range_.from_ - 1U) &&
           rs_.is_cancelled());
  return *this;
}

frun::iterator frun::iterator::operator--(int) {
  auto r = *this;
  --(*this);
  return r;
}

bool frun::iterator::operator==(iterator const o) const { return rs_ == o.rs_; }

bool frun::iterator::operator!=(iterator o) const { return !(*this == o); }

run_stop frun::iterator::operator*() const { return rs_; }

frun::frun(timetable const& tt, rt_timetable const* rtt, run r)
    : run{r}, tt_{&tt}, rtt_{rtt} {
  if (!is_rt() && rtt != nullptr) {
    rt_ = rtt->resolve_rt(r.t_);
  }
  if (!is_scheduled() && rtt != nullptr &&
      r.rt_ != rt_transport_idx_t::invalid()) {
    t_ = rtt->resolve_static(r.rt_);
  }
}

std::string_view frun::name() const {
  if (is_rt() && rtt_ != nullptr) {
    return rtt_->transport_name(*tt_, rt_);
  }
  if (is_scheduled()) {
    return tt_->transport_name(t_.t_idx_);
  }
  return "";
}

debug frun::dbg() const {
  if (is_rt() && rtt_ != nullptr) {
    return rtt_->dbg(*tt_, rt_);
  }
  if (is_scheduled()) {
    return tt_->dbg(t_.t_idx_);
  }
  return debug{};
}

stop_idx_t frun::first_valid(stop_idx_t const from) const {
  for (auto i = from; i != stop_range_.to_; ++i) {
    if (operator[](i - stop_range_.from_).in_allowed() ||
        operator[](i - stop_range_.from_).out_allowed()) {
      return i;
    }
  }
  log(log_lvl::error, "frun", "no first valid found: id={}, name={}, dbg={}",
      fmt::streamed(id()), name(), fmt::streamed(dbg()));
  return stop_range_.to_;
}

stop_idx_t frun::last_valid() const {
  auto n = stop_range_.size();
  for (auto r = 0; r != n; ++r) {
    auto const i = static_cast<stop_idx_t>(stop_range_.to_ - r - 1);
    if (operator[](i - stop_range_.from_).in_allowed() ||
        operator[](i - stop_range_.from_).out_allowed()) {
      return i;
    }
  }
  log(log_lvl::error, "frun", "no last valid found: id={}, name={}, dbg={}",
      fmt::streamed(id()), name(), fmt::streamed(dbg()));
  return stop_range_.to_;
}

frun::iterator frun::begin() const {
  return iterator{
      run_stop{.fr_ = this, .stop_idx_ = first_valid(stop_range_.from_)}};
}

frun::iterator frun::end() const {
  return iterator{run_stop{.fr_ = this, .stop_idx_ = stop_range_.to_}};
}

frun::iterator begin(frun const& fr) { return fr.begin(); }
frun::iterator end(frun const& fr) { return fr.end(); }

std::reverse_iterator<frun::iterator> frun::rbegin() const {
  return std::make_reverse_iterator(end());
}

std::reverse_iterator<frun::iterator> frun::rend() const {
  return std::make_reverse_iterator(begin());
}

std::reverse_iterator<frun::iterator> rbegin(frun const& fr) {
  return fr.rbegin();
}
std::reverse_iterator<frun::iterator> rend(frun const& fr) { return fr.rend(); }

stop_idx_t frun::size() const {
  return static_cast<stop_idx_t>(
      (is_rt() && rtt_ != nullptr)
          ? rtt_->rt_transport_location_seq_[rt_].size()
      : is_scheduled()
          ? tt_->route_location_seq_[tt_->transport_route_[t_.t_idx_]].size()
          : 0U);
}

run_stop frun::operator[](stop_idx_t const i) const {
  return run_stop{this, static_cast<stop_idx_t>(stop_range_.from_ + i)};
}

clasz frun::get_clasz() const {
  if (is_rt() && rtt_ != nullptr) {
    return rtt_->rt_transport_clasz_.at(rt_);
  } else {
    return tt_->route_clasz_.at(tt_->transport_route_.at(t_.t_idx_));
  }
}

clasz frun::get_scheduled_clasz() const {
  if (!is_scheduled()) {
    return clasz::kOther;
  }
  return tt_->route_clasz_.at(tt_->transport_route_.at(t_.t_idx_));
}

trip_idx_t frun::trip_idx() const {
  utl::verify(is_scheduled(), "can't get trip_idx for unscheduled trip");
  return tt_->transport_trip_.at(t_.t_idx_);
}

route_color frun::get_route_color() const {
  if (!is_scheduled()) {
    return route_color{};
  }
  auto const color_sections =
      tt_->transport_section_route_colors_.at(fr_->t_.t_idx_);
  return color_sections.at(color_sections.size() == 1U ? 0U
                                                       : section_idx(ev_type));
}

provider const& frun::get_provider() const {
  return is_scheduled() ? tt_->get_provider(t_.t_idx_) : tt_->providers_.at({});
}

std::string_view frun::trip_display_name() const {
  if (is_rt() && rtt_ != nullptr) {
    return rtt_->transport_name(*tt_, rt_);
  }
  if (is_scheduled()) {
    return tt_->trip_display_names_[trip_idx()].view();
  }
  return "?";
}

trip_id frun::id() const {
  if (is_scheduled()) {
    auto const trip_idx = frun::trip_idx();
    auto const trip_id_idx = tt_->trip_ids_[trip_idx].at(0);
    return {tt_->trip_id_strings_[trip_id_idx].view(),
            tt_->trip_id_src_[trip_id_idx]};
  } else if (rtt_ != nullptr &&
             holds_alternative<rt_add_trip_id_idx_t>(
                 rtt_->rt_transport_static_transport_[rt_])) {
    auto const add_idx =
        rtt_->rt_transport_static_transport_[rt_].as<rt_add_trip_id_idx_t>();
    return {rtt_->additional_trip_ids_.get(add_idx),
            rtt_->rt_transport_src_[rt_]};
  } else {
    return {};
  }
}

bool frun::bikes_allowed() const {
  if (is_rt() && rtt_ != nullptr) {
    return rtt_->rt_transport_bikes_allowed_.test(rt_);
  } else {
    return tt_->route_bikes_allowed_.test(tt_->transport_route_.at(t_.t_idx_));
  }
}

bool frun::is_cancelled() const {
  if (rtt_ == nullptr) {
    return false;
  }
  if (is_rt()) {
    return rtt_->rt_transport_is_cancelled_[rt_];
  }
  if (is_scheduled()) {
    return !rtt_->bitfields_[rtt_->transport_traffic_days_[t_.t_idx_]].test(
        to_idx(t_.day_));
  }
  return false;
}

void run_stop::print(std::ostream& out,
                     bool const first,
                     bool const last) const {
  auto const& tz = tt().locations_.timezones_.at(
      get_transport_stop_tz(*fr_->tt_, fr_->t_.t_idx_, get_location().l_));

  // Print stop index, location name.
  fmt::print(out, "  {:2}: {:7} {:.<48}", stop_idx_, get_location().id_,
             name());

  // Print arrival (or whitespace if there's none).
  if (!first && stop_idx_ != fr_->stop_range_.from_) {
    auto const scheduled = scheduled_time(event_type::kArr);
    auto const rt = time(event_type::kArr);
    fmt::print(out, "{}a: {} [{}]", (out_allowed() ? ' ' : '-'),
               date::format("%d.%m %R", scheduled),
               date::format("%d.%m %R", to_local_time(tz, scheduled)));
    if (fr_->is_rt() && rtt() != nullptr) {  // RT if available.
      fmt::print(out, "  RT {} [{}]", date::format("%d.%m %R", rt),
                 date::format("%d.%m %R", to_local_time(tz, rt)));
    }
  } else if (fr_->is_rt() && rtt() != nullptr) {
    // Skipped w/ RT info.
    fmt::print(out, "                            ");
    fmt::print(out, "                               ");
  } else {
    // Skipped w/o RT info.
    fmt::print(out, "                             ");
  }

  // Print departure (or whitespace if there's none).
  if (!last && stop_idx_ != fr_->stop_range_.to_ - 1U) {
    fmt::print(out, " ");
    auto const scheduled = scheduled_time(event_type::kDep);
    auto const rt = time(event_type::kDep);
    fmt::print(out, "{}d: {} [{}]", (in_allowed() ? ' ' : '-'),
               date::format("%d.%m %R", scheduled),
               date::format("%d.%m %R", to_local_time(tz, scheduled)));
    if (fr_->is_rt() && rtt() != nullptr) {  // RT if available.
      fmt::print(out, "  RT {} [{}]", date::format("%d.%m %R", rt),
                 date::format("%d.%m %R", to_local_time(tz, rt)));
    }
  }

  // Print trip info.
  if (fr_->is_scheduled() && !last && stop_idx_ != fr_->stop_range_.to_ - 1U) {
    auto const& tt = *fr_->tt_;
    out << "  [";
    auto const trip_idx = tt.transport_trip_[fr_->t_.t_idx_];
    for (auto const [dbg, id] :
         utl::zip(tt.trip_debug_.at(trip_idx), tt.trip_ids_.at(trip_idx))) {
      out << "{name=" << tt.trip_display_names_.at(trip_idx).view() << ", day=";
      date::to_stream(
          out, "%F",
          tt.internal_interval_days().from_ + to_idx(fr_->t_.day_) * 1_days);
      out << ", id=" << tt.trip_id_strings_.at(id).view()
          << ", src=" << static_cast<int>(to_idx(tt.trip_id_src_.at(id)));
      out << "}";
    }
    out << "]";
  }
}

std::ostream& operator<<(std::ostream& out, run_stop const& stp) {
  stp.print(out);
  return out;
}

std::ostream& operator<<(std::ostream& out, frun const& fr) {
  for (auto const stp : fr) {
    out << stp << "\n";
  }
  return out;
}

frun frun::from_rt(timetable const& tt,
                   rt_timetable const* rtt,
                   rt_transport_idx_t const rt_t) {
  auto const to =
      static_cast<stop_idx_t>(rtt->rt_transport_location_seq_[rt_t].size());
  return {tt, rtt, {.stop_range_ = {stop_idx_t{0U}, to}, .rt_ = rt_t}};
}

frun frun::from_t(timetable const& tt,
                  rt_timetable const* rtt,
                  transport const t) {
  auto const to = static_cast<stop_idx_t>(
      tt.route_location_seq_[tt.transport_route_[t.t_idx_]].size());
  return {tt, rtt, {.t_ = t, .stop_range_ = {stop_idx_t{0U}, to}}};
}

}  // namespace nigiri::rt
