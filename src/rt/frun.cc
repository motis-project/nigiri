#include "nigiri/rt/frun.h"

#include <algorithm>
#include <iterator>

#include "nigiri/lookup/get_transport_stop_tz.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::rt {

stop frun::run_stop::get_stop() const noexcept {
  return stop{
      (fr_->is_rt() && rtt() != nullptr)
          ? rtt()->rt_transport_location_seq_[fr_->rt_][stop_idx_]
          : tt().route_location_seq_[tt().transport_route_[fr_->t_.t_idx_]]
                                    [stop_idx_]};
}

std::string_view frun::run_stop::name() const noexcept {
  auto const l = get_location_idx();
  auto const type = tt().locations_.types_.at(l);
  auto const p =
      (type == location_type::kGeneratedTrack || type == location_type::kTrack)
          ? tt().locations_.parents_.at(l)
          : l;
  return tt().locations_.names_.at(p).view();
}

std::string_view frun::run_stop::id() const noexcept {
  auto const l = get_location_idx();
  auto const type = tt().locations_.types_.at(l);
  return tt()
      .locations_.ids_
      .at(type == location_type::kGeneratedTrack
              ? tt().locations_.parents_.at(l)
              : l)
      .view();
}

std::string_view frun::run_stop::track() const noexcept {
  auto const l = get_location_idx();
  return (tt().locations_.types_.at(l) == location_type::kTrack ||
          tt().locations_.types_.at(l) == location_type::kGeneratedTrack)
             ? tt().locations_.names_.at(l).view()
             : "";
}

location frun::run_stop::get_location() const noexcept {
  assert(fr_->size() > stop_idx_);
  return location{*fr_->tt_, get_stop().location_idx()};
}

geo::latlng frun::run_stop::pos() const noexcept {
  assert(fr_->size() > stop_idx_);
  return fr_->tt_->locations_.coordinates_[get_stop().location_idx()];
}

location_idx_t frun::run_stop::get_location_idx() const noexcept {
  assert(fr_->size() > stop_idx_);
  return get_stop().location_idx();
}

unixtime_t frun::run_stop::scheduled_time(
    event_type const ev_type) const noexcept {
  assert(fr_->size() > stop_idx_);
  return fr_->is_scheduled()
             ? tt().event_time(fr_->t_, stop_idx_, ev_type)
             : rtt()->unix_event_time(fr_->rt_, stop_idx_, ev_type);
}

unixtime_t frun::run_stop::time(event_type const ev_type) const noexcept {
  assert(fr_->size() > stop_idx_);
  return (fr_->is_rt() && rtt() != nullptr)
             ? rtt()->unix_event_time(fr_->rt_, stop_idx_, ev_type)
             : tt().event_time(fr_->t_, stop_idx_, ev_type);
}

duration_t frun::run_stop::delay(event_type const ev_type) const noexcept {
  assert(fr_->size() > stop_idx_);
  return time(ev_type) - scheduled_time(ev_type);
}

trip_idx_t frun::run_stop::get_trip_idx(
    event_type const ev_type) const noexcept {
  auto const sections = tt().transport_to_trip_section_.at(fr_->t_.t_idx_);
  return tt()
      .merged_trips_[sections.at(sections.size() == 1U ? 0U
                                                       : section_idx(ev_type))]
      .at(0);
}

std::string_view frun::run_stop::trip_display_name(
    event_type const ev_type) const noexcept {
  return tt().trip_display_names_[get_trip_idx(ev_type)].view();
}

stop_idx_t frun::run_stop::section_idx(
    event_type const ev_type) const noexcept {
  return static_cast<stop_idx_t>(ev_type == event_type::kDep ? stop_idx_
                                                             : stop_idx_ - 1);
}

std::string_view frun::run_stop::line(event_type const ev_type) const noexcept {
  if (fr_->is_rt() && rtt() != nullptr) {
    auto const rt_line = rtt()->rt_transport_line_.at(fr_->rt_);
    return rt_line.empty() ? scheduled_line(ev_type) : rt_line.view();
  } else {
    return scheduled_line(ev_type);
  }
}

provider const& frun::run_stop::get_provider(
    event_type const ev_type) const noexcept {
  auto const provider_sections =
      tt().transport_section_providers_.at(fr_->t_.t_idx_);
  auto const provider_idx = provider_sections.at(
      provider_sections.size() == 1U ? 0U : section_idx(ev_type));
  return tt().providers_.at(provider_idx);
}

std::string_view frun::run_stop::direction(
    event_type const ev_type) const noexcept {
  if (!fr_->is_scheduled()) {
    return "";
  }

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
  return direction;
}

std::string_view frun::run_stop::scheduled_line(
    event_type const ev_type) const noexcept {
  if (!fr_->is_scheduled()) {
    return "";
  }

  auto const section_lines = tt().transport_section_lines_.at(fr_->t_.t_idx_);
  if (section_lines.empty()) {
    return "";
  } else {
    auto const line_idx = section_lines.size() == 1U
                              ? section_lines[0]
                              : section_lines.at(section_idx(ev_type));
    return tt().trip_lines_.at(line_idx).view();
  }
}

clasz frun::run_stop::get_clasz(event_type const ev_type) const noexcept {
  if (fr_->is_rt() && rtt() != nullptr) {
    auto const clasz_sections = rtt()->rt_transport_section_clasz_.at(fr_->rt_);
    return clasz_sections.at(
        clasz_sections.size() == 1U ? 0U : section_idx(ev_type));
  } else {
    auto const clasz_sections =
        tt().route_section_clasz_.at(tt().transport_route_.at(fr_->t_.t_idx_));
    return clasz_sections.at(
        clasz_sections.size() == 1U ? 0U : section_idx(ev_type));
  }
}

clasz frun::run_stop::get_scheduled_clasz(
    event_type const ev_type) const noexcept {
  if (!fr_->is_scheduled()) {
    return clasz();
  }
  auto const clasz_sections =
      tt().route_section_clasz_.at(tt().transport_route_.at(fr_->t_.t_idx_));
  return clasz_sections.at(clasz_sections.size() == 1U ? 0U
                                                       : section_idx(ev_type));
}

bool frun::run_stop::bikes_allowed(event_type const ev_type) const noexcept {
  if (fr_->is_rt() && rtt() != nullptr) {
    auto const bikes_allowed_seq =
        rtt()->rt_bikes_allowed_per_section_.at(fr_->rt_);
    return bikes_allowed_seq.at(
        bikes_allowed_seq.size() == 1U ? 0U : section_idx(ev_type));
  } else {
    auto const bikes_allowed_seq = tt().route_bikes_allowed_per_section_.at(
        tt().transport_route_.at(fr_->t_.t_idx_));
    return bikes_allowed_seq.at(
        bikes_allowed_seq.size() == 1U ? 0U : section_idx(ev_type));
  }
}

route_color frun::run_stop::get_route_color(event_type ev_type) const noexcept {
  auto const color_sections =
      tt().transport_section_route_colors_.at(fr_->t_.t_idx_);
  return color_sections.at(color_sections.size() == 1U ? 0U
                                                       : section_idx(ev_type));
}

bool frun::run_stop::is_canceled() const noexcept {
  return get_stop().is_cancelled();
}

bool frun::run_stop::in_allowed() const noexcept {
  return get_stop().in_allowed();
}

bool frun::run_stop::out_allowed() const noexcept {
  return get_stop().out_allowed();
}

bool frun::run_stop::in_allowed_wheelchair() const noexcept {
  return get_stop().in_allowed_wheelchair();
}

bool frun::run_stop::out_allowed_wheelchair() const noexcept {
  return get_stop().out_allowed_wheelchair();
}

bool frun::run_stop::in_allowed(bool const wheelchair) const noexcept {
  return wheelchair ? in_allowed_wheelchair() : in_allowed();
}

bool frun::run_stop::out_allowed(bool const wheelchair) const noexcept {
  return wheelchair ? out_allowed_wheelchair() : out_allowed();
}

timetable const& frun::run_stop::tt() const noexcept { return *fr_->tt_; }
rt_timetable const* frun::run_stop::rtt() const noexcept { return fr_->rtt_; }

frun::iterator& frun::iterator::operator++() noexcept {
  do {
    ++rs_.stop_idx_;
  } while (rs_.stop_idx_ != rs_.fr_->stop_range_.to_ && rs_.is_canceled());
  return *this;
}

frun::iterator frun::iterator::operator++(int) noexcept {
  auto r = *this;
  ++(*this);
  return r;
}

frun::iterator& frun::iterator::operator--() noexcept {
  do {
    --rs_.stop_idx_;
  } while (rs_.stop_idx_ !=
               static_cast<stop_idx_t>(rs_.fr_->stop_range_.from_ - 1U) &&
           rs_.is_canceled());
  return *this;
}

frun::iterator frun::iterator::operator--(int) noexcept {
  auto r = *this;
  --(*this);
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
    : run{r}, tt_{&tt}, rtt_{rtt} {
  if (!is_rt() && rtt != nullptr) {
    rt_ = rtt->resolve_rt(r.t_);
  }
  if (!is_scheduled() && rtt != nullptr &&
      r.rt_ != rt_transport_idx_t::invalid()) {
    t_ = rtt->resolve_static(r.rt_);
  }
}

std::string_view frun::name() const noexcept {
  return (is_rt() && rtt_ != nullptr) ? rtt_->transport_name(*tt_, rt_)
                                      : tt_->transport_name(t_.t_idx_);
}

debug frun::dbg() const noexcept {
  return (is_rt() && rtt_ != nullptr) ? rtt_->dbg(*tt_, rt_)
                                      : tt_->dbg(t_.t_idx_);
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

frun::iterator frun::begin() const noexcept {
  return iterator{
      run_stop{.fr_ = this, .stop_idx_ = first_valid(stop_range_.from_)}};
}

frun::iterator frun::end() const noexcept {
  return iterator{run_stop{.fr_ = this, .stop_idx_ = stop_range_.to_}};
}

frun::iterator begin(frun const& fr) noexcept { return fr.begin(); }
frun::iterator end(frun const& fr) noexcept { return fr.end(); }

std::reverse_iterator<frun::iterator> frun::rbegin() const noexcept {
  return std::make_reverse_iterator(end());
}

std::reverse_iterator<frun::iterator> frun::rend() const noexcept {
  return std::make_reverse_iterator(begin());
}

std::reverse_iterator<frun::iterator> rbegin(frun const& fr) noexcept {
  return fr.rbegin();
}
std::reverse_iterator<frun::iterator> rend(frun const& fr) noexcept {
  return fr.rend();
}

stop_idx_t frun::size() const noexcept {
  return static_cast<stop_idx_t>(
      (is_rt() && rtt_ != nullptr)
          ? rtt_->rt_transport_location_seq_[rt_].size()
          : tt_->route_location_seq_[tt_->transport_route_[t_.t_idx_]].size());
}

frun::run_stop frun::operator[](stop_idx_t const i) const noexcept {
  return run_stop{this, static_cast<stop_idx_t>(stop_range_.from_ + i)};
}

clasz frun::get_clasz() const noexcept {
  if (is_scheduled()) {
    return tt_->route_section_clasz_[tt_->transport_route_[t_.t_idx_]].at(0);
  } else {
    return rtt_->rt_transport_section_clasz_[rt_].at(0);
  }
}

void frun::for_each_shape_point(
    shapes_storage const* const shapes_data,
    interval<stop_idx_t> const& range,
    std::function<void(geo::latlng const&)> const& callback) const {
  auto const process_stops = [&](interval<stop_idx_t> const subrange) {
    for (auto const stop_index : subrange) {
      auto const coordinate = (*this)[stop_index].pos();
      callback(coordinate);
    }
  };
  // Full fallback
  if (shapes_data == nullptr) {
    process_stops(range);
    return;
  }
  // Helper functions
  auto const shift_interval = [](interval<stop_idx_t> const& range_interval,
                                 int const offset) {
    assert(range_interval.from_ >= -offset);  // Result interval must be valid
    return interval{static_cast<stop_idx_t>(range_interval.from_ + offset),
                    static_cast<stop_idx_t>(range_interval.to_ + offset)};
  };
  auto const get_first_offset = [&](trip_idx_t const trip_index) {
    auto const range_offset =
        static_cast<stop_idx_t>(stop_range_.from_ + range.from_);
    if (range_offset == stop_idx_t{0}) {
      return 0;
    }
    auto const candidates = interval{stop_idx_t{0}, range_offset};
    auto const first = std::find_if(
        std::begin(candidates), std::end(candidates),
        [&](stop_idx_t const candidate) {
          auto const previous_stop = (candidate < range.from_)
                                         ? (*this)[static_cast<stop_idx_t>(
                                               range.from_ - (candidate + 1))]
                                         : (*this)[-static_cast<stop_idx_t>(
                                               (candidate + 1) - range.from_)];
          return previous_stop.get_trip_idx(event_type::kDep) != trip_index;
        });
    if (first == std::end(candidates)) {
      return static_cast<int>(stop_range_.from_);
    } else {
      return *first - range.from_;
    }
  };
  auto const process_shape = [&](std::span<geo::latlng const> const shape) {
    if (shape.empty()) {
      return;
    }
    std::for_each(std::begin(shape), std::end(shape), callback);
  };
  auto const process_trip_stops = [&](stop_idx_t const from,
                                      trip_idx_t const current_trip_index) {
    auto stop_index = from;
    // Reminder: Always at least 2 stops
    auto run_stop = (*this)[stop_index];
    auto next_trip_index = trip_idx_t{0};
    callback(run_stop.pos());
    do {
      run_stop = (*this)[++stop_index];
      next_trip_index = run_stop.get_trip_idx(event_type::kDep);
      callback(run_stop.pos());
    } while (next_trip_index == current_trip_index);
    return std::make_pair(next_trip_index, stop_index - from);
  };
  // Setup
  assert(range.from_ < range.to_);
  assert(stop_range_.from_ + range.to_ <= stop_range_.to_ + 1);
  auto const from = (*this)[range.from_];
  auto const to = (*this)[range.to_ - 1];
  auto const final_trip_index = to.get_trip_idx(event_type::kArr);
  auto current_trip_index = from.get_trip_idx(event_type::kDep);
  auto current_interval =
      shift_interval(range, get_first_offset(current_trip_index));
  auto current_offset = range.from_;
  // Process trips, excluding last
  while (current_trip_index != final_trip_index) {
    auto const [shape, stops] = shapes_data->get_shape_with_stop_count(
        current_trip_index, current_interval.from_);
    auto offset_adjustment = stops - 1;
    if (stops > 0) {
      process_shape(shape);
      current_trip_index =
          (*this)[static_cast<stop_idx_t>(current_offset + offset_adjustment)]
              .get_trip_idx(event_type::kDep);
    } else {
      std::tie(current_trip_index, offset_adjustment) =
          process_trip_stops(stop_idx_t{current_offset}, current_trip_index);
    }
    current_offset += offset_adjustment;
    current_interval =
        interval{stop_idx_t{0}, static_cast<stop_idx_t>(current_interval.to_ -
                                                        current_interval.from_ -
                                                        offset_adjustment)};
  }
  // Final trip
  auto const shape =
      shapes_data->get_shape(current_trip_index, current_interval);
  if (!shape.empty()) {
    process_shape(shape);
  } else {
    process_stops(interval{stop_idx_t{current_offset}, range.to_});
  }
}

trip_id frun::id() const noexcept {
  if (is_scheduled()) {
    auto const trip_idx =
        tt_->merged_trips_[tt_->transport_to_trip_section_.at(t_.t_idx_).at(0)]
            .at(0);
    auto const trip_id_idx = tt_->trip_ids_[trip_idx].at(0);
    return {tt_->trip_id_strings_[trip_id_idx].view(),
            tt_->trip_id_src_[trip_id_idx]};
  } else if (holds_alternative<rt_add_trip_id_idx_t>(
                 rtt_->rt_transport_static_transport_[rt_])) {
    auto const add_idx =
        rtt_->rt_transport_static_transport_[rt_].as<rt_add_trip_id_idx_t>();
    return {rtt_->trip_id_strings_[add_idx].view(),
            rtt_->rt_transport_src_[rt_]};
  } else {
    return {};
  }
}

trip_idx_t frun::trip_idx() const {
  if (is_scheduled()) {
    return tt_
        ->merged_trips_[tt_->transport_to_trip_section_.at(t_.t_idx_).at(0)]
        .at(0);
  }
  throw utl::fail("trip idx only for scheduled trip");
}

void frun::run_stop::print(std::ostream& out,
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
    auto const& trip_section = tt.transport_to_trip_section_.at(fr_->t_.t_idx_);
    auto const& merged_trips = tt.merged_trips_.at(
        trip_section.size() == 1U ? trip_section[0]
                                  : trip_section.at(stop_idx_));

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
        date::to_stream(
            out, "%F",
            tt.internal_interval_days().from_ + to_idx(fr_->t_.day_) * 1_days);
        out << ", id=" << tt.trip_id_strings_.at(id).view()
            << ", src=" << static_cast<int>(to_idx(tt.trip_id_src_.at(id)));
        out << "}";
      }
    }
    out << "]";
  }
}

std::ostream& operator<<(std::ostream& out, frun::run_stop const& stp) {
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
                   rt_transport_idx_t const rt_t) noexcept {
  auto const to =
      static_cast<stop_idx_t>(rtt->rt_transport_location_seq_[rt_t].size());
  return {tt, rtt, {.stop_range_ = {stop_idx_t{0U}, to}, .rt_ = rt_t}};
}

frun frun::from_t(timetable const& tt,
                  rt_timetable const* rtt,
                  transport const t) noexcept {
  auto const to = static_cast<stop_idx_t>(
      tt.route_location_seq_[tt.transport_route_[t.t_idx_]].size());
  return {tt, rtt, {.t_ = t, .stop_range_ = {stop_idx_t{0U}, to}}};
}

}  // namespace nigiri::rt
