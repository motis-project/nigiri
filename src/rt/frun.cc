#include "nigiri/rt/frun.h"

#include <iterator>
#include <optional>
#include <span>
#include <variant>

#include "utl/overloaded.h"
#include "utl/verify.h"
#include "utl/visit.h"
#include "utl/zip.h"

#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"

namespace nigiri::rt {

constexpr auto const kUnknownProvider =
    provider{.id_ = string_idx_t::invalid(),
             .name_ = kEmptyTranslation,
             .url_ = kEmptyTranslation,
             .src_ = source_idx_t::invalid()};

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

std::string_view run_stop::name(lang_t const& lang) const {
  auto const l = get_location_idx();
  auto const p = tt().locations_.parents_.at(l);
  auto const x = p == location_idx_t::invalid() ? l : p;
  return tt().translate(lang, tt().locations_.names_.at(x));
}

std::string_view run_stop::id() const {
  auto const l = get_location_idx();
  auto const p = tt().locations_.parents_.at(l);
  auto const x = p == location_idx_t::invalid() ? l : p;
  return tt().locations_.ids_.at(x).view();
}

std::pair<date::sys_days, duration_t> run_stop::get_trip_start(
    event_type const ev_type) const {
  if (!fr_->is_scheduled()) {
    // additional trip - return first departure
    auto first = [&]() {
      auto copy = *this;
      copy.stop_idx_ = 0U;
      return copy;
    }();
    auto const first_dep = first.time(event_type::kDep);
    auto const day =
        std::chrono::time_point_cast<date::sys_days::duration>(first_dep);
    return {day, first_dep - day};
  } else {
    // go to first trip stop
    auto first_trip_stop = [&]() {
      auto const trip = get_trip_idx(ev_type);
      auto copy = *this;
      while (copy.stop_idx_ > 0 &&
             copy.get_trip_idx(event_type::kArr) == trip) {
        --copy.stop_idx_;
      }
      return copy;
    }();

    // service date + start time
    auto const [static_transport, utc_start_day] = fr_->t_;
    auto const [first_dep_offset, tz_offset] =
        tt().transport_first_dep_offset_[static_transport].to_offset();
    auto const utc_dep =
        tt().event_mam(static_transport, first_trip_stop.stop_idx_,
                       event_type::kDep)
            .as_duration();
    auto const gtfs_static_dep = utc_dep + first_dep_offset + tz_offset;

    auto const day =
        (tt().internal_interval_days().from_ +
         std::chrono::days{to_idx(utc_start_day)} - first_dep_offset);

    return {day, gtfs_static_dep};
  }
}

std::string_view run_stop::track(lang_t const& lang) const {
  auto const l = get_location_idx();
  return tt().translate(lang, tt().locations_.platform_codes_.at(l));
}

geo::latlng run_stop::pos() const {
  assert(fr_->size() > stop_idx_);
  return fr_->tt_->locations_.coordinates_[get_stop().location_idx()];
}

loc run_stop::get_loc() const { return {tt(), get_location_idx()}; }

location_idx_t run_stop::get_location_idx() const {
  assert(fr_->size() > stop_idx_);
  return get_stop().location_idx();
}

std::string_view run_stop::get_location_id() const {
  return tt().locations_.ids_[get_location_idx()].view();
}

location_idx_t run_stop::get_scheduled_location_idx() const {
  assert(fr_->size() > stop_idx_);
  return get_scheduled_stop().location_idx();
}

run_stop run_stop::get_last_trip_stop(event_type const ev_type) const {
  auto const end = fr_->size();
  if (!fr_->is_scheduled()) {
    return run_stop{fr_, static_cast<stop_idx_t>(end - 1)};
  }

  auto const trip = get_trip_idx(ev_type);
  auto copy = *this;
  if (copy.stop_idx_ == end - 1) {
    return copy;
  }

  // Can't be (end-1), so ++stop_idx is fine.
  ++copy.stop_idx_;

  // Can't be 0 after ++stop_idx, so get_trip_idx(kArr) is fine.
  while (copy.stop_idx_ < end - 1 &&
         copy.get_trip_idx(event_type::kArr) == trip) {
    ++copy.stop_idx_;
  }

  if (copy.get_trip_idx(event_type::kArr) != trip) {
    copy.stop_idx_ -= 1;
  }

  return copy;
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

timezone_idx_t run_stop::get_tz(event_type const ev_type) const {
  auto const location_tz =
      tt().locations_.location_timezones_.at(get_location_idx());
  if (location_tz != timezone_idx_t::invalid()) {
    return location_tz;
  }

  auto const& p = get_provider(ev_type);
  if (p.tz_ != timezone_idx_t::invalid()) {
    return p.tz_;
  }

  if (fr_->is_rt() && fr_->rtt_ != nullptr) {
    auto const src_idx = rtt()->rt_transport_src_.at(fr_->rt_);
    auto const it = std::lower_bound(
        begin(tt().providers_), end(tt().providers_), src_idx,
        [&](provider const& a, source_idx_t const b) { return a.src_ < b; });
    if (it != end(tt().providers_) && it->src_ == src_idx &&
        it->tz_ != timezone_idx_t::invalid()) {
      return it->tz_;
    }
  }
  return timezone_idx_t::invalid();
}

std::optional<std::string> run_stop::get_tz_name(
    event_type const ev_type) const {
  auto const tz_idx = get_tz(ev_type);
  if (tz_idx == timezone_idx_t::invalid()) {
    return std::nullopt;
  }
  auto const& tz = fr_->tt_->timezones_.at(tz_idx);
  auto const* date_tz = to_time_zone(tz);
  return date_tz == nullptr ? std::nullopt : std::optional{date_tz->name()};
}

trip_idx_t run_stop::get_trip_idx(event_type const ev_type) const {
  utl::verify(fr_->t_.t_idx_ != transport_idx_t::invalid(),
              "can't get trip_idx for unscheduled trip");
  auto const sections = tt().transport_to_trip_section_.at(fr_->t_.t_idx_);
  return tt()
      .merged_trips_[sections.at(sections.size() == 1U ? 0U
                                                       : section_idx(ev_type))]
      .at(0);
}

std::pair<timetable::route_ids const*, route_id_idx_t> run_stop::get_route(
    event_type const ev_type) const {
  if (fr_->is_scheduled()) {
    auto const trip_idx = get_trip_idx(ev_type);
    auto const r =
        tt().trip_route_id_[trip_idx] == route_id_idx_t::invalid()
            ? nullptr
            : &tt().route_ids_
                   [tt().trip_id_src_[tt().trip_ids_[trip_idx].front()]];
    return {r, tt().trip_route_id_[trip_idx]};
  } else if (auto const route_id = rtt()->rt_transport_route_id_[fr_->rt_];
             route_id != route_id_idx_t::invalid()) {
    auto const& r = tt().route_ids_[rtt()->rt_transport_src_[fr_->rt_]];
    return std::pair{&r, route_id};
  } else {
    return {nullptr, route_id_idx_t::invalid()};
  }
}

std::string_view run_stop::get_route_id(event_type const ev_type) const {
  auto const [route_ids, route_id_idx] = get_route(ev_type);
  return route_ids == nullptr ? "?" : route_ids->ids_.get(route_id_idx);
}

direction_id_t run_stop::get_direction_id(event_type const ev_type) const {
  if (fr_->is_scheduled()) {
    auto const trip_idx = get_trip_idx(ev_type);
    return direction_id_t{tt().trip_direction_id_.test(trip_idx)};
  } else {
    return direction_id_t{rtt()->rt_transport_direction_id_[fr_->rt_] ? 1 : 0};
  }
}

std::optional<route_type_t> run_stop::route_type(
    event_type const ev_type) const {
  auto const [route_ids, route_id_idx] = get_route(ev_type);
  return route_ids == nullptr
             ? std::nullopt
             : std::optional{route_ids->route_id_type_.at(route_id_idx)};
}

std::string_view run_stop::route_short_name(event_type const ev_type,
                                            lang_t const& lang) const {
  auto const [route_ids, route_id_idx] = get_route(ev_type);
  return route_ids == nullptr
             ? "?"
             : tt().translate(
                   lang, route_ids->route_id_short_names_.at(route_id_idx));
}

std::string_view run_stop::route_long_name(event_type const ev_type,
                                           lang_t const& lang) const {
  auto const [route_ids, route_id_idx] = get_route(ev_type);
  return route_ids == nullptr
             ? "?"
             : tt().translate(lang,
                              route_ids->route_id_long_names_.at(route_id_idx));
}

std::string_view run_stop::trip_short_name(event_type const ev_type,
                                           lang_t const& lang) const {
  if (fr_->is_scheduled()) {
    return tt().translate(lang, tt().trip_short_names_[get_trip_idx(ev_type)]);
  } else {
    return utl::visit(
        rtt()->trip_short_name(tt(), fr_->rt_),
        [&](translation_idx_t const t) { return tt().translate(lang, t); },
        [](std::string_view s) { return s; });
  }
}

std::string_view run_stop::display_name(event_type const ev_type,
                                        lang_t const& lang) const {
  if (fr_->is_scheduled()) {
    return tt().translate(lang,
                          tt().trip_display_names_[get_trip_idx(ev_type)]);
  }
  auto const name = route_short_name(ev_type, lang);
  return name.empty() ? trip_short_name(ev_type, lang) : name;
}

stop_idx_t run_stop::section_idx(event_type const ev_type) const {
  return static_cast<stop_idx_t>(ev_type == event_type::kDep ? stop_idx_
                                                             : stop_idx_ - 1);
}

provider_idx_t run_stop::get_provider_idx(event_type const ev_type) const {
  auto const [route_ids, route_id_idx] = get_route(ev_type);
  return route_ids == nullptr ? provider_idx_t::invalid()
                              : route_ids->route_id_provider_.at(route_id_idx);
}

provider const& run_stop::get_provider(event_type const ev_type) const {
  auto const provider_idx = get_provider_idx(ev_type);
  if (provider_idx != provider_idx_t::invalid()) {
    return tt().providers_.at(provider_idx);
  }
  return kUnknownProvider;
}

std::string_view run_stop::direction(lang_t const& lang,
                                     event_type const ev_type) const {
  if (fr_->is_scheduled()) {
    auto const direction_sections =
        tt().transport_section_directions_.at(fr_->t_.t_idx_);
    if (!direction_sections.empty()) {
      auto const direction_idx =
          direction_sections.size() == 1U
              ? direction_sections.at(0)
              : direction_sections.at(section_idx(ev_type));
      if (direction_idx != kEmptyTranslation) {
        return tt().translate(lang, direction_idx);
      }
    }
  }
  if (rtt() != nullptr || fr_->is_scheduled()) {
    return run_stop{.fr_ = fr_,
                    .stop_idx_ = static_cast<stop_idx_t>(fr_->size() - 1U)}
        .name(lang);
  }
  return "";
}

attribute_combination_idx_t run_stop::get_attribute_combination(
    event_type ev_type) const {
  if (!fr_->is_scheduled()) {
    return attribute_combination_idx_t{0};
  }
  auto const attribute_sections =
      tt().transport_section_attributes_[fr_->t_.t_idx_];
  return attribute_sections.empty()
             ? attribute_combination_idx_t{0}
             : attribute_sections.at(
                   attribute_sections.size() == 1U ? 0U : section_idx(ev_type));
}

clasz run_stop::get_clasz(event_type const ev_type) const {
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

clasz run_stop::get_scheduled_clasz(event_type const ev_type) const {
  if (!fr_->is_scheduled()) {
    return clasz::kOther;
  }
  auto const clasz_sections =
      tt().route_section_clasz_.at(tt().transport_route_.at(fr_->t_.t_idx_));
  return clasz_sections.at(clasz_sections.size() == 1U ? 0U
                                                       : section_idx(ev_type));
}

bool run_stop::bikes_allowed(event_type const ev_type) const {
  if (fr_->is_rt() && rtt() != nullptr) {
    if (rtt()->rt_transport_bikes_allowed_[to_idx(fr_->rt_) * 2U]) {
      return true;
    } else if (!rtt()
                    ->rt_transport_bikes_allowed_[to_idx(fr_->rt_) * 2U + 1U]) {
      return false;
    } else {
      auto const bikes_allowed_seq =
          rtt()->rt_bikes_allowed_per_section_.at(fr_->rt_);
      return bikes_allowed_seq.at(
          bikes_allowed_seq.size() == 1U ? 0U : section_idx(ev_type));
    }
  } else {
    auto const r = tt().transport_route_.at(fr_->t_.t_idx_);
    if (tt().route_bikes_allowed_[to_idx(r) * 2U]) {
      return true;
    } else if (!tt().route_bikes_allowed_[to_idx(r) * 2U + 1U]) {
      return false;
    } else {
      auto const bikes_allowed_seq = tt().route_bikes_allowed_per_section_.at(
          tt().transport_route_.at(fr_->t_.t_idx_));
      return bikes_allowed_seq.at(
          bikes_allowed_seq.size() == 1U ? 0U : section_idx(ev_type));
    }
  }
}

bool run_stop::cars_allowed(event_type const ev_type) const {
  if (fr_->is_rt() && rtt() != nullptr) {
    if (rtt()->rt_transport_cars_allowed_[to_idx(fr_->rt_) * 2U]) {
      return true;
    } else if (!rtt()->rt_transport_cars_allowed_[to_idx(fr_->rt_) * 2U + 1U]) {
      return false;
    } else {
      auto const cars_allowed_seq =
          rtt()->rt_cars_allowed_per_section_.at(fr_->rt_);
      return cars_allowed_seq.at(
          cars_allowed_seq.size() == 1U ? 0U : section_idx(ev_type));
    }
  } else {
    auto const r = tt().transport_route_.at(fr_->t_.t_idx_);
    if (tt().route_cars_allowed_[to_idx(r) * 2U]) {
      return true;
    } else if (!tt().route_cars_allowed_[to_idx(r) * 2U + 1U]) {
      return false;
    } else {
      auto const cars_allowed_seq = tt().route_cars_allowed_per_section_.at(
          tt().transport_route_.at(fr_->t_.t_idx_));
      return cars_allowed_seq.at(
          cars_allowed_seq.size() == 1U ? 0U : section_idx(ev_type));
    }
  }
}

route_color run_stop::get_route_color(event_type ev_type) const {
  auto const [routes, route_id_idx] = get_route(ev_type);
  return routes == nullptr
             ? route_color{.color_ = color_t{0}, .text_color_ = color_t{0}}
             : routes->route_id_colors_[route_id_idx];
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

std::string_view frun::name(lang_t const& lang) const {
  if (is_rt() && rtt_ != nullptr) {
    return operator[](0U).route_short_name(
        stop_range_.from_ == size() - 1U ? event_type::kArr : event_type::kDep,
        lang);
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
      fmt::streamed(id()), name({}), fmt::streamed(dbg()));
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
      fmt::streamed(id()), name({}), fmt::streamed(dbg()));
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
  if (is_scheduled()) {
    return tt_->route_section_clasz_[tt_->transport_route_[t_.t_idx_]].at(0);
  }
  if (rtt_ != nullptr) {
    return rtt_->rt_transport_section_clasz_[rt_].at(0);
  }
  return clasz::kOther;
}

void frun::for_each_trip(
    std::function<void(trip_idx_t const, interval<stop_idx_t> const)> const&
        callback) const {
  if (t_.t_idx_ == transport_idx_t::invalid()) {
    callback(trip_idx_t::invalid(), stop_range_);
    return;
  }
  auto curr_trip_idx = trip_idx_t::invalid();
  auto curr_from = stop_idx_t{0U};
  for (auto const [from, to] :
       utl::pairwise(interval{stop_idx_t{0U}, stop_range_.to_})) {
    auto const trip_idx =
        (*this)[static_cast<stop_idx_t>(from - stop_range_.from_)]  //
            .get_trip_idx(event_type::kDep);
    if (trip_idx == curr_trip_idx) {
      continue;
    }
    if (from != 0U) {
      callback(curr_trip_idx, interval{curr_from, to});
    }
    curr_trip_idx = trip_idx;
    curr_from = from;
  }
  callback(curr_trip_idx, interval{curr_from, stop_range_.to_});
}

void frun::for_each_shape_point(
    shapes_storage const* shapes_data,
    interval<stop_idx_t> const range,
    std::function<void(geo::latlng const&)> const& callback) const {
  utl::verify(range.size() >= 2, "Range must contain at least 2 stops. Is {}",
              range.size());
  assert(stop_range_.from_ + range.to_ <= stop_range_.to_);
  auto const absolute_stop_range = range >> stop_range_.from_;
  auto const get_subshape = [&](interval<stop_idx_t> absolute_range,
                                trip_idx_t const trip_idx,
                                stop_idx_t const absolute_trip_offset)
      -> std::variant<std::span<geo::latlng const>, interval<stop_idx_t>> {
    if (shapes_data != nullptr && trip_idx != trip_idx_t::invalid()) {
      auto const shape = shapes_data->get_shape(
          trip_idx, absolute_range << absolute_trip_offset);
      if (!shape.empty()) {
        return shape;
      }
    }
    return absolute_range << stop_range_.from_;
  };
  auto start_pos = (*this)[range.from_].pos();
  callback(start_pos);
  auto consume_pos = [&, last_pos = std::move(start_pos), changed = false](
                         geo::latlng const& pos,
                         bool const force_if_unchanged = false) mutable {
    if (pos != last_pos || (force_if_unchanged && !changed)) {
      callback(pos);
      changed = true;
    }
    last_pos = pos;
  };
  for_each_trip([&](trip_idx_t const trip_idx,
                    interval<stop_idx_t> const subrange) {
    auto const common_stops = subrange.intersect(absolute_stop_range);
    if (common_stops.size() > 1) {
      std::visit(utl::overloaded{[&](std::span<geo::latlng const> shape) {
                                   for (auto const& pos : shape) {
                                     consume_pos(pos);
                                   }
                                 },
                                 [&](interval<stop_idx_t> relative_range) {
                                   for (auto const stop_idx : relative_range) {
                                     consume_pos((*this)[stop_idx].pos());
                                   }
                                 }},
                 get_subshape(common_stops, trip_idx, subrange.from_));
    }
  });
  consume_pos((*this)[static_cast<stop_idx_t>(range.to_ - 1)].pos(), true);
}

trip_id frun::id() const {
  if (is_scheduled()) {
    auto const trip_idx =
        tt_->merged_trips_[tt_->transport_to_trip_section_.at(t_.t_idx_).at(0)]
            .at(0);
    auto const trip_id_idx = tt_->trip_ids_[trip_idx].at(0);
    return {tt_->trip_id_strings_[trip_id_idx].view(),
            tt_->trip_id_src_[trip_id_idx]};
  } else if (rtt_ != nullptr &&
             holds_alternative<rt_add_trip_id_idx_t>(
                 rtt_->rt_transport_static_transport_[rt_])) {
    auto const add_idx =
        rtt_->rt_transport_static_transport_[rt_].as<rt_add_trip_id_idx_t>();
    auto const src = rtt_->rt_transport_src_[rt_];
    return {rtt_->additional_trips_.at(src).ids_.get(add_idx), src};
  } else {
    return {};
  }
}

bool frun::is_cancelled() const {
  if (rtt_ == nullptr) {
    return false;
  }
  if (is_rt()) {
    return rtt_->rt_transport_is_cancelled_[to_idx(rt_)];
  }
  if (is_scheduled()) {
    return !rtt_->bitfields_[rtt_->transport_traffic_days_[t_.t_idx_]].test(
        to_idx(t_.day_));
  }
  return false;
}

trip_idx_t frun::trip_idx() const {
  if (is_scheduled()) {
    return tt_
        ->merged_trips_[tt_->transport_to_trip_section_.at(t_.t_idx_).at(0)]
        .at(0);
  }
  throw utl::fail("trip idx only for scheduled trip");
}

void run_stop::print(std::ostream& out,
                     bool const first,
                     bool const last) const {
  auto const& tz =
      tt().timezones_.at(get_tz(last ? event_type::kArr : event_type::kDep));

  // Print stop index, location name.
  fmt::print(out, "  {:2}: {:7} {:.<48}", stop_idx_, get_location_id(),
             name({}));

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
        out << "{name="
            << display_name(last ? event_type::kArr : event_type::kDep, {})
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

std::ostream& operator<<(std::ostream& out, run_stop const& stp) {
  stp.print(out, stp.stop_idx_ == 0U, stp.stop_idx_ == stp.fr_->size() - 1);
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
