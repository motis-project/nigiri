#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

#include "date/date.h"

#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"
#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "nigiri/abi.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/loader.h"
#include "nigiri/loader/hrd/loader.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/logging.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/frun.h"

#include "nigiri/common/interval.h"
#include "cista/memory_holder.h"

using namespace date;

struct nigiri_timetable {
  std::shared_ptr<nigiri::timetable> tt;
  std::shared_ptr<nigiri::rt_timetable> rtt;
};

nigiri_timetable_t* nigiri_load_from_dir(nigiri::loader::dir const& d,
                                         int64_t from_ts,
                                         int64_t to_ts,
                                         unsigned link_stop_distance) {
  auto loaders =
      std::vector<std::unique_ptr<nigiri::loader::loader_interface>>{};
  loaders.emplace_back(std::make_unique<nigiri::loader::gtfs::gtfs_loader>());
  loaders.emplace_back(
      std::make_unique<nigiri::loader::hrd::hrd_5_00_8_loader>());
  loaders.emplace_back(
      std::make_unique<nigiri::loader::hrd::hrd_5_20_26_loader>());
  loaders.emplace_back(
      std::make_unique<nigiri::loader::hrd::hrd_5_20_39_loader>());
  loaders.emplace_back(
      std::make_unique<nigiri::loader::hrd::hrd_5_20_avv_loader>());

  auto const src = nigiri::source_idx_t{0U};

  auto const c =
      utl::find_if(loaders, [&](auto&& l) { return l->applicable(d); });
  utl::verify(c != end(loaders), "no loader applicable to the given file(s)");
  nigiri::log(nigiri::log_lvl::info, "main",
              "loading nigiri timetable with configuration {}", (*c)->name());

  auto t = new nigiri_timetable_t;
  t->tt = std::make_unique<nigiri::timetable>();

  t->tt->n_sources_ = 1U;
  t->tt->date_range_ = {floor<days>(std::chrono::system_clock::from_time_t(
                            static_cast<time_t>(from_ts))),
                        floor<days>(std::chrono::system_clock::from_time_t(
                            static_cast<time_t>(to_ts)))};

  nigiri::loader::register_special_stations(*t->tt);

  auto local_bitfield_indices =
      nigiri::hash_map<nigiri::bitfield, nigiri::bitfield_idx_t>{};
  (*c)->load({.link_stop_distance_ = link_stop_distance,
              .default_tz_ = "Europe/Berlin"},
             src, d, *t->tt, local_bitfield_indices, nullptr, nullptr);
  nigiri::loader::finalize(*t->tt);

  t->rtt = std::make_shared<nigiri::rt_timetable>(
      nigiri::rt::create_rt_timetable(*t->tt, t->tt->date_range_.from_));
  return t;
}

nigiri_timetable_t* nigiri_load(char const* path,
                                int64_t from_ts,
                                int64_t to_ts) {
  return nigiri_load_linking_stops(path, from_ts, to_ts, 0);
}

nigiri_timetable_t* nigiri_load_linking_stops(char const* path,
                                              int64_t from_ts,
                                              int64_t to_ts,
                                              unsigned link_stop_distance) {
  auto const progress_tracker = utl::activate_progress_tracker("libnigiri");
  auto const silencer = utl::global_progress_bars{true};

  auto const tt_path = std::filesystem::path{path};
  auto const d = nigiri::loader::make_dir(tt_path);
  return nigiri_load_from_dir(*d, from_ts, to_ts, link_stop_distance);
}

void nigiri_destroy(nigiri_timetable_t const* t) { delete t; }

int64_t nigiri_get_start_day_ts(nigiri_timetable_t const* t) {
  return std::chrono::system_clock::to_time_t(
      t->tt->internal_interval_days().from_);
}

uint16_t nigiri_get_day_count(nigiri_timetable_t const* t) {
  return static_cast<uint16_t>(t->tt->internal_interval_days().size().count());
}

uint32_t nigiri_get_transport_count(nigiri_timetable_t const* t) {
  return t->tt->transport_route_.size();
}

nigiri_transport_t* nigiri_get_transport(nigiri_timetable_t const* t,
                                         uint32_t idx) {
  auto const tidx = nigiri::transport_idx_t{idx};
  auto transport = new nigiri_transport_t;

  auto route_idx = t->tt->transport_route_[tidx];

  auto n_stops = t->tt->route_location_seq_[route_idx].size();

  auto event_mams = new int16_t[(n_stops - 1) * 2];
  for (size_t i = 0; i < n_stops; i++) {
    if (i != 0) {
      event_mams[i * 2 - 1] =
          t->tt
              ->event_mam(tidx, static_cast<nigiri::stop_idx_t>(i),
                          nigiri::event_type::kArr)
              .count();
    }
    if (i != n_stops - 1) {
      event_mams[i * 2] =
          t->tt
              ->event_mam(tidx, static_cast<nigiri::stop_idx_t>(i),
                          nigiri::event_type::kDep)
              .count();
    }
  }
  transport->route_idx = static_cast<nigiri::route_idx_t::value_t>(route_idx);
  transport->n_event_mams = (static_cast<uint16_t>(n_stops) - 1) * 2;
  transport->event_mams = event_mams;
  transport->name = t->tt->transport_name(tidx).data();
  transport->name_len =
      static_cast<uint32_t>(t->tt->transport_name(tidx).length());
  return transport;
}

void nigiri_destroy_transport(nigiri_transport_t const* transport) {
  delete[] transport->event_mams;
  delete transport;
}

bool nigiri_is_transport_active(nigiri_timetable_t const* t,
                                uint32_t const transport_idx,
                                uint16_t day_idx) {
  auto const tidx = nigiri::transport_idx_t{transport_idx};
  return t->tt->bitfields_[t->tt->transport_traffic_days_[tidx]].test(day_idx);
}

uint32_t nigiri_get_route_count(nigiri_timetable_t const* t) {
  return t->tt->n_routes();
}

nigiri_route_t* nigiri_get_route(nigiri_timetable_t const* t, uint32_t idx) {
  auto const ridx = nigiri::route_idx_t{idx};
  auto stops = t->tt->route_location_seq_[ridx];
  auto route = new nigiri_route_t;
  auto const n_stops = stops.size();
  route->stops = new nigiri_route_stop_t[n_stops];
  if (n_stops > 0) {
    std::memcpy(route->stops, &stops.front(),
                sizeof(nigiri_route_stop_t) * n_stops);
  }
  route->n_stops = static_cast<uint16_t>(n_stops);
  route->clasz =
      static_cast<uint16_t>(t->tt->route_section_clasz_[ridx].front());
  return route;
}

void nigiri_destroy_route(nigiri_route_t const* route) {
  delete[] route->stops;
  delete route;
}

uint32_t nigiri_get_location_count(nigiri_timetable_t const* t) {
  return t->tt->n_locations();
}

nigiri_location_t* nigiri_get_location_with_footpaths(
    nigiri_timetable_t const* t, uint32_t idx, bool incoming_footpaths) {
  auto const l = nigiri::location_idx_t{idx};
  auto location = new nigiri_location_t;

  auto const name = t->tt->get_default_translation(t->tt->locations_.names_[l]);
  location->name = name.data();
  location->name_len = static_cast<uint32_t>(name.length());

  auto const id = t->tt->locations_.ids_[l].view();
  location->id = id.data();
  location->id_len = static_cast<uint32_t>(id.length());

  auto const pos = t->tt->locations_.coordinates_[l];
  location->lat = pos.lat_;
  location->lon = pos.lng_;

  location->transfer_time =
      static_cast<uint16_t>(t->tt->locations_.transfer_time_[l].count());
  auto footpaths = incoming_footpaths ? t->tt->locations_.footpaths_in_[0][l]
                                      : t->tt->locations_.footpaths_out_[0][l];
  auto const n_footpaths = footpaths.size();
  location->footpaths = new nigiri_footpath_t[n_footpaths];
  if (n_footpaths > 0) {
    std::memcpy(location->footpaths, &footpaths.front(),
                sizeof(nigiri_footpath_t) * n_footpaths);
  }
  location->n_footpaths = static_cast<uint32_t>(n_footpaths);

  auto const parent = t->tt->locations_.parents_[l];
  location->parent =
      t->tt->locations_.parents_[l] == nigiri::location_idx_t::invalid()
          ? 0
          : static_cast<nigiri::location_idx_t::value_t>(parent);
  return location;
}

nigiri_location_t* nigiri_get_location(nigiri_timetable_t const* t,
                                       uint32_t idx) {
  return nigiri_get_location_with_footpaths(t, idx, false);
}

void nigiri_destroy_location(nigiri_location_t const* location) {
  delete[] location->footpaths;
  delete location;
}

void nigiri_update_with_rt_from_buf(nigiri_timetable_t const* t,
                                    std::string_view protobuf,
                                    void (*callback)(nigiri_event_change_t,
                                                     void* context),
                                    void* context) {
  auto const src = nigiri::source_idx_t{0U};
  auto const tag = "";

  auto const rtt_callback =
      [&](nigiri::transport const transport, nigiri::stop_idx_t const stop_idx,
          nigiri::event_type const ev_type,
          std::optional<nigiri::location_idx_t> const location_idx,
          std::optional<bool> in_out_allowed,
          std::optional<nigiri::duration_t> const delay) {
        nigiri_event_change_t const c = {
            .transport_idx =
                static_cast<nigiri::transport_idx_t::value_t>(transport.t_idx_),
            .day_idx = static_cast<nigiri::day_idx_t::value_t>(transport.day_),
            .stop_idx = stop_idx,
            .is_departure = ev_type != nigiri::event_type::kArr,
            .stop_change = !delay.has_value(),
            .stop_location_idx = static_cast<nigiri::location_idx_t::value_t>(
                location_idx.value_or(nigiri::location_idx_t::invalid())),
            .stop_in_out_allowed = in_out_allowed.value_or(true),
            .delay = delay.value_or(nigiri::duration_t{0}).count()};
        callback(c, context);
      };

  t->rtt->set_change_callback(rtt_callback);
  try {
    nigiri::rt::gtfsrt_update_buf(*t->tt, *t->rtt, src, tag, protobuf);
  } catch (std::exception const& e) {
    nigiri::log(nigiri::log_lvl::error, "main",
                "GTFS-RT update error (tag={}) {}", tag, e.what());
  } catch (...) {
    nigiri::log(nigiri::log_lvl::error, "main",
                "Unknown GTFS-RT update error (tag={})", tag);
  }
  t->rtt->reset_change_callback();
}

void nigiri_update_with_rt(nigiri_timetable_t const* t,
                           char const* gtfsrt_pb_path,
                           void (*callback)(nigiri_event_change_t,
                                            void* context),
                           void* context) {
  auto const file = cista::mmap{gtfsrt_pb_path, cista::mmap::protection::READ};
  return nigiri_update_with_rt_from_buf(t, file.view(), callback, context);
}

nigiri::pareto_set<nigiri::routing::journey> raptor_search(
    nigiri::timetable const& tt,
    nigiri::rt_timetable const* rtt,
    nigiri::routing::query q,
    bool backward_search) {
  static auto search_state = nigiri::routing::search_state{};
  static auto algo_state = nigiri::routing::raptor_state{};
  if (backward_search) {
    using algo_t =
        nigiri::routing::raptor<nigiri::direction::kBackward, true, 0,
                                nigiri::routing::search_mode::kOneToOne>;
    return *(nigiri::routing::search<nigiri::direction::kBackward, algo_t>{
        tt, rtt, search_state, algo_state, std::move(q)}
                 .execute()
                 .journeys_);
  } else {
    using algo_t =
        nigiri::routing::raptor<nigiri::direction::kForward, true, 0,
                                nigiri::routing::search_mode::kOneToOne>;
    return *(nigiri::routing::search<nigiri::direction::kForward, algo_t>{
        tt, rtt, search_state, algo_state, std::move(q)}
                 .execute()
                 .journeys_);
  }
}

nigiri_pareto_set_t* nigiri_get_journeys(nigiri_timetable_t const* t,
                                         uint32_t start_location_idx,
                                         uint32_t destination_location_idx,
                                         int64_t time,
                                         bool backward_search) {
  using namespace date;
  using namespace nigiri;
  auto q = nigiri::routing::query{
      .start_time_ = floor<std::chrono::minutes>(
          std::chrono::system_clock::from_time_t(static_cast<time_t>(time))),
      .start_ = {{nigiri::location_idx_t{start_location_idx}, 0_minutes, 0U}},
      .destination_ = {{nigiri::location_idx_t{destination_location_idx},
                        0_minutes, 0U}},
      .prf_idx_ = 0};

  auto journeys = raptor_search(*t->tt, t->rtt.get(), q, backward_search);
  auto const n_journeys =
      static_cast<std::size_t>(std::distance(journeys.begin(), journeys.end()));
  auto js = new nigiri_journey_t[n_journeys];

  auto const pareto_set = new nigiri_pareto_set_t;
  pareto_set->n_journeys = static_cast<uint16_t>(n_journeys);
  pareto_set->journeys = js;

  auto i = 0;
  for (auto it = journeys.begin(); it != journeys.end(); it++, i++) {
    js[i].n_legs = static_cast<uint16_t>(it->legs_.size());
    js[i].legs = new nigiri_leg_t[it->legs_.size()];
    js[i].start_time = std::chrono::system_clock::to_time_t(it->start_time_);
    js[i].dest_time = std::chrono::system_clock::to_time_t(it->dest_time_);

    for (auto const [j, leg] : utl::enumerate(it->legs_)) {
      auto const l = &js[i].legs[j];

      auto const set_run =
          [&](nigiri::routing::journey::run_enter_exit const& run) {
            auto const frun = nigiri::rt::frun{*t->tt, t->rtt.get(), run.r_};
            auto const from = frun[run.stop_range_.from_];
            auto const to = frun[run.stop_range_.to_ - 1U];
            l->is_footpath = false;
            l->transport_idx =
                run.r_.is_scheduled()
                    ? static_cast<nigiri::transport_idx_t::value_t>(
                          run.r_.t_.t_idx_)
                    : 0;
            l->day_idx =
                run.r_.is_scheduled()
                    ? static_cast<nigiri::day_idx_t::value_t>(run.r_.t_.day_)
                    : 0;
            l->from_stop_idx = run.stop_range_.from_;
            l->from_location_idx = static_cast<nigiri::location_idx_t::value_t>(
                from.get_location_idx());
            l->to_stop_idx = run.stop_range_.to_ - 1U;
            l->to_location_idx = static_cast<nigiri::location_idx_t::value_t>(
                to.get_location_idx());
            l->duration =
                static_cast<uint32_t>((to.time(nigiri::event_type::kArr) -
                                       from.time(nigiri::event_type::kDep))
                                          .count());
          };
      auto const set_footpath = [&, leg](nigiri::footpath const fp) {
        l->is_footpath = true;
        l->transport_idx = 0;
        l->day_idx = 0;
        l->from_stop_idx = 0;
        l->from_location_idx =
            static_cast<nigiri::location_idx_t::value_t>(leg.from_);
        l->to_stop_idx = 0;
        l->to_location_idx =
            static_cast<nigiri::location_idx_t::value_t>(leg.to_);
        l->duration = static_cast<uint32_t>(fp.duration().count());
      };
      auto const set_offset = [&, leg](nigiri::routing::offset const x) {
        l->is_footpath = true;
        l->transport_idx = 0;
        l->day_idx = 0;
        l->from_stop_idx = 0;
        l->from_location_idx =
            static_cast<nigiri::location_idx_t::value_t>(leg.from_);
        l->to_stop_idx = 0;
        l->to_location_idx =
            static_cast<nigiri::location_idx_t::value_t>(leg.to_);
        l->duration = static_cast<uint32_t>(x.duration().count());
      };
      std::visit(utl::overloaded{set_run, set_footpath, set_offset}, leg.uses_);
    }
  }
  return pareto_set;
}

void nigiri_destroy_journeys(nigiri_pareto_set_t const* journeys) {
  for (int i = 0; i < journeys->n_journeys; i++) {
    delete[] journeys->journeys[i].legs;
  }
  delete[] journeys->journeys;
  delete journeys;
}

// --- Extended accessors ---

int64_t nigiri_get_external_interval_start(nigiri_timetable_t const* t) {
  return std::chrono::system_clock::to_time_t(
      std::chrono::time_point_cast<std::chrono::seconds>(
          t->tt->date_range_.from_));
}

int64_t nigiri_get_external_interval_end(nigiri_timetable_t const* t) {
  return std::chrono::system_clock::to_time_t(
      std::chrono::time_point_cast<std::chrono::seconds>(
          t->tt->date_range_.to_));
}

uint16_t nigiri_get_route_stop_count(nigiri_timetable_t const* t,
                                     uint32_t route_idx) {
  auto const ridx = nigiri::route_idx_t{route_idx};
  return static_cast<uint16_t>(t->tt->route_location_seq_[ridx].size());
}

uint32_t nigiri_get_transport_name(nigiri_timetable_t const* t,
                                   uint32_t transport_idx,
                                   char* buf,
                                   uint32_t buf_len) {
  auto const tidx = nigiri::transport_idx_t{transport_idx};
  auto const name = t->tt->transport_name(tidx);
  auto const len = static_cast<uint32_t>(name.length());
  if (buf != nullptr && buf_len > 0) {
    auto const copy_len = std::min(len, buf_len);
    std::memcpy(buf, name.data(), copy_len);
  }
  return len;
}

uint32_t nigiri_get_transport_route(nigiri_timetable_t const* t,
                                    uint32_t transport_idx) {
  auto const tidx = nigiri::transport_idx_t{transport_idx};
  return static_cast<nigiri::route_idx_t::value_t>(
      t->tt->transport_route_[tidx]);
}

uint32_t nigiri_get_rt_transport_count(nigiri_timetable_t const* t) {
  return t->rtt ? t->rtt->n_rt_transports() : 0U;
}

uint32_t nigiri_find_location(nigiri_timetable_t const* t,
                              char const* id,
                              uint32_t id_len) {
  auto const src = nigiri::source_idx_t{0U};
  auto const result = t->tt->find(
      nigiri::location_id{std::string_view{id, id_len}, src});
  return result.has_value()
             ? static_cast<nigiri::location_idx_t::value_t>(*result)
             : UINT32_MAX;
}

uint16_t nigiri_to_day_idx(nigiri_timetable_t const* t, int64_t unix_ts) {
  auto const ut = nigiri::unixtime_t{
      std::chrono::duration_cast<nigiri::i32_minutes>(
          std::chrono::system_clock::from_time_t(static_cast<time_t>(unix_ts))
              .time_since_epoch())};
  auto const [d, m] = t->tt->day_idx_mam(ut);
  return static_cast<uint16_t>(cista::to_idx(d));
}

int64_t nigiri_to_unixtime(nigiri_timetable_t const* t,
                           uint16_t day_idx,
                           uint16_t minutes_after_midnight) {
  auto const ut = t->tt->to_unixtime(
      nigiri::day_idx_t{day_idx},
      nigiri::minutes_after_midnight_t{minutes_after_midnight});
  return std::chrono::system_clock::to_time_t(
      std::chrono::time_point_cast<std::chrono::seconds>(
          nigiri::unixtime_t{} + ut.time_since_epoch()));
}

// --- Phase 1: Detail accessors ---

uint32_t nigiri_get_source_count(nigiri_timetable_t const* t) {
  return static_cast<uint32_t>(t->tt->n_sources());
}

bool nigiri_get_location_detail(nigiri_timetable_t const* t,
                                uint32_t location_idx,
                                nigiri_location_detail_t* out) {
  if (location_idx >= t->tt->n_locations()) {
    return false;
  }
  auto const l = nigiri::location_idx_t{location_idx};

  auto const pos = t->tt->locations_.coordinates_[l];
  out->lat = pos.lat_;
  out->lon = pos.lng_;

  auto const name = t->tt->get_default_translation(t->tt->locations_.names_[l]);
  out->name = name.data();
  out->name_len = static_cast<uint32_t>(name.length());

  auto const id = t->tt->locations_.ids_[l].view();
  out->id = id.data();
  out->id_len = static_cast<uint32_t>(id.length());

  out->location_type =
      static_cast<uint8_t>(t->tt->locations_.types_[l]);

  auto const parent = t->tt->locations_.parents_[l];
  out->parent_idx =
      parent == nigiri::location_idx_t::invalid()
          ? UINT32_MAX
          : static_cast<nigiri::location_idx_t::value_t>(parent);

  out->src_idx =
      static_cast<nigiri::source_idx_t::value_t>(t->tt->locations_.src_[l]);

  out->transfer_time =
      static_cast<uint16_t>(t->tt->locations_.transfer_time_[l].count());

  return true;
}

bool nigiri_get_route_detail(nigiri_timetable_t const* t,
                             uint32_t route_idx,
                             nigiri_route_detail_t* out) {
  if (route_idx >= t->tt->n_routes()) {
    return false;
  }
  auto const ridx = nigiri::route_idx_t{route_idx};

  // Default: clasz from route
  out->clasz = static_cast<uint8_t>(t->tt->route_clasz_[ridx]);

  // Initialize string fields to empty
  out->short_name = "";
  out->short_name_len = 0;
  out->long_name = "";
  out->long_name_len = 0;
  out->agency_name = "";
  out->agency_name_len = 0;
  out->agency_id = "";
  out->agency_id_len = 0;
  out->color = 0;
  out->text_color = 0;

  // Navigate: route → first transport → first trip → route_id_idx → names
  auto const transport_range = t->tt->route_transport_ranges_[ridx];
  if (transport_range.from_ == transport_range.to_) {
    return true;  // route with no transports — return with defaults
  }

  auto const first_transport = transport_range.from_;
  auto const trip_sections =
      t->tt->transport_to_trip_section_[first_transport];
  if (trip_sections.empty()) {
    return true;
  }

  auto const merged = t->tt->merged_trips_[trip_sections.front()];
  if (merged.empty()) {
    return true;
  }

  auto const trip_idx = merged.front();
  auto const route_id_idx = t->tt->trip_route_id_[trip_idx];

  // Find the source for this transport's locations
  auto const first_loc_seq = t->tt->route_location_seq_[ridx];
  auto src = nigiri::source_idx_t{0U};
  if (!first_loc_seq.empty()) {
    auto const first_loc =
        nigiri::stop{first_loc_seq.front()}.location_idx();
    src = t->tt->locations_.src_[first_loc];
  }

  if (static_cast<uint32_t>(cista::to_idx(src)) < t->tt->route_ids_.size()) {
    auto const& rids = t->tt->route_ids_[src];

    if (static_cast<uint32_t>(cista::to_idx(route_id_idx)) <
        rids.route_id_short_names_.size()) {
      auto const sn =
          t->tt->get_default_translation(rids.route_id_short_names_[route_id_idx]);
      out->short_name = sn.data();
      out->short_name_len = static_cast<uint32_t>(sn.length());
    }

    if (static_cast<uint32_t>(cista::to_idx(route_id_idx)) <
        rids.route_id_long_names_.size()) {
      auto const ln =
          t->tt->get_default_translation(rids.route_id_long_names_[route_id_idx]);
      out->long_name = ln.data();
      out->long_name_len = static_cast<uint32_t>(ln.length());
    }

    if (static_cast<uint32_t>(cista::to_idx(route_id_idx)) <
        rids.route_id_colors_.size()) {
      auto const colors = rids.route_id_colors_[route_id_idx];
      out->color = static_cast<uint32_t>(colors.color_);
      out->text_color = static_cast<uint32_t>(colors.text_color_);
    }

    if (static_cast<uint32_t>(cista::to_idx(route_id_idx)) <
        rids.route_id_provider_.size()) {
      auto const provider_idx = rids.route_id_provider_[route_id_idx];
      if (static_cast<uint32_t>(cista::to_idx(provider_idx)) <
          t->tt->providers_.size()) {
        auto const& prov = t->tt->providers_[provider_idx];
        auto const prov_name = t->tt->get_default_translation(prov.name_);
        out->agency_name = prov_name.data();
        out->agency_name_len = static_cast<uint32_t>(prov_name.length());

        auto const prov_id = t->tt->strings_.get(prov.id_);
        out->agency_id = prov_id.data();
        out->agency_id_len = static_cast<uint32_t>(prov_id.length());
      }
    }
  }

  return true;
}

// --- Phase 2: Tag lookup support ---

bool nigiri_get_transport_trip_id(nigiri_timetable_t const* t,
                                  uint32_t transport_idx,
                                  char const** id_out,
                                  uint32_t* id_len_out) {
  if (transport_idx >= t->tt->transport_route_.size()) {
    return false;
  }
  auto const tidx = nigiri::transport_idx_t{transport_idx};
  auto const trip_sections = t->tt->transport_to_trip_section_[tidx];
  if (trip_sections.empty()) {
    return false;
  }
  auto const merged = t->tt->merged_trips_[trip_sections.front()];
  if (merged.empty()) {
    return false;
  }
  auto const trip_idx = merged.front();
  auto const trip_id_indices = t->tt->trip_ids_[trip_idx];
  if (trip_id_indices.empty()) {
    return false;
  }
  auto const trip_id_str =
      t->tt->trip_id_strings_[trip_id_indices.front()].view();
  *id_out = trip_id_str.data();
  *id_len_out = static_cast<uint32_t>(trip_id_str.length());
  return true;
}

uint32_t nigiri_get_transport_source(nigiri_timetable_t const* t,
                                     uint32_t transport_idx) {
  if (transport_idx >= t->tt->transport_route_.size()) {
    return UINT32_MAX;
  }
  auto const tidx = nigiri::transport_idx_t{transport_idx};
  auto const trip_sections = t->tt->transport_to_trip_section_[tidx];
  if (trip_sections.empty()) {
    return UINT32_MAX;
  }
  auto const merged = t->tt->merged_trips_[trip_sections.front()];
  if (merged.empty()) {
    return UINT32_MAX;
  }
  auto const trip_idx = merged.front();
  auto const trip_id_indices = t->tt->trip_ids_[trip_idx];
  if (trip_id_indices.empty()) {
    return UINT32_MAX;
  }
  return static_cast<nigiri::source_idx_t::value_t>(
      t->tt->trip_id_src_[trip_id_indices.front()]);
}

int16_t nigiri_get_transport_first_dep_mam(nigiri_timetable_t const* t,
                                           uint32_t transport_idx) {
  if (transport_idx >= t->tt->transport_route_.size()) {
    return -1;
  }
  auto const tidx = nigiri::transport_idx_t{transport_idx};
  return t->tt
      ->event_mam(tidx, nigiri::stop_idx_t{0}, nigiri::event_type::kDep)
      .count();
}

bool nigiri_get_route_gtfs_id(nigiri_timetable_t const* t,
                              uint32_t route_idx,
                              char const** id_out,
                              uint32_t* id_len_out) {
  if (route_idx >= t->tt->n_routes()) {
    return false;
  }
  auto const ridx = nigiri::route_idx_t{route_idx};
  auto const transport_range = t->tt->route_transport_ranges_[ridx];
  if (transport_range.from_ == transport_range.to_) {
    return false;
  }
  auto const first_transport = transport_range.from_;
  auto const trip_sections =
      t->tt->transport_to_trip_section_[first_transport];
  if (trip_sections.empty()) {
    return false;
  }
  auto const merged = t->tt->merged_trips_[trip_sections.front()];
  if (merged.empty()) {
    return false;
  }
  auto const trip_idx = merged.front();
  auto const route_id_idx = t->tt->trip_route_id_[trip_idx];

  auto const first_loc_seq = t->tt->route_location_seq_[ridx];
  auto src = nigiri::source_idx_t{0U};
  if (!first_loc_seq.empty()) {
    auto const first_loc =
        nigiri::stop{first_loc_seq.front()}.location_idx();
    src = t->tt->locations_.src_[first_loc];
  }

  if (static_cast<uint32_t>(cista::to_idx(src)) >=
      t->tt->route_ids_.size()) {
    return false;
  }
  auto const& rids = t->tt->route_ids_[src];
  if (static_cast<uint32_t>(cista::to_idx(route_id_idx)) >=
      rids.ids_.size()) {
    return false;
  }
  auto const id_str = rids.ids_.get(route_id_idx);
  *id_out = id_str.data();
  *id_len_out = static_cast<uint32_t>(id_str.length());
  return true;
}

bool nigiri_day_to_date_str(nigiri_timetable_t const* t,
                            uint16_t day_idx,
                            char* buf_out) {
  auto const days_from_epoch =
      t->tt->internal_interval_days().from_ + date::days{day_idx};
  auto const ymd = date::year_month_day{days_from_epoch};
  auto const y = static_cast<int>(ymd.year());
  auto const m = static_cast<unsigned>(ymd.month());
  auto const d = static_cast<unsigned>(ymd.day());
  // buf_out must be at least 9 bytes (8 chars + null terminator).
  // We write into a local buffer to satisfy the compiler's truncation checks.
  char tmp[16];
  std::snprintf(tmp, sizeof(tmp), "%04d%02u%02u", y, m, d);
  std::memcpy(buf_out, tmp, 8);
  buf_out[8] = '\0';
  return true;
}

// --- Phase 3: Routing support ---

uint32_t nigiri_get_location_routes(nigiri_timetable_t const* t,
                                    uint32_t location_idx,
                                    uint32_t* routes_out,
                                    uint32_t max_routes) {
  if (location_idx >= t->tt->n_locations()) {
    return 0;
  }
  auto const l = nigiri::location_idx_t{location_idx};
  auto const routes = t->tt->location_routes_[l];
  auto const count = static_cast<uint32_t>(routes.size());
  if (routes_out != nullptr) {
    auto const n = std::min(count, max_routes);
    for (uint32_t i = 0; i < n; ++i) {
      routes_out[i] =
          static_cast<nigiri::route_idx_t::value_t>(routes[i]);
    }
    return n;
  }
  return count;
}

uint16_t nigiri_get_stop_idx_in_route(nigiri_timetable_t const* t,
                                      uint32_t route_idx,
                                      uint32_t location_idx) {
  if (route_idx >= t->tt->n_routes()) {
    return UINT16_MAX;
  }
  auto const ridx = nigiri::route_idx_t{route_idx};
  auto const seq = t->tt->route_location_seq_[ridx];
  for (uint16_t i = 0; i < seq.size(); ++i) {
    auto const stop_loc =
        nigiri::stop{seq[i]}.location_idx();
    if (static_cast<nigiri::location_idx_t::value_t>(stop_loc) ==
        location_idx) {
      return i;
    }
  }
  return UINT16_MAX;
}

int16_t nigiri_get_event_mam(nigiri_timetable_t const* t,
                             uint32_t transport_idx,
                             uint16_t stop_idx,
                             bool is_arrival) {
  if (transport_idx >= t->tt->transport_route_.size()) {
    return -1;
  }
  auto const tidx = nigiri::transport_idx_t{transport_idx};
  auto const ev = is_arrival ? nigiri::event_type::kArr
                             : nigiri::event_type::kDep;
  return t->tt->event_mam(tidx, nigiri::stop_idx_t{stop_idx}, ev).count();
}

bool nigiri_get_route_transport_range(nigiri_timetable_t const* t,
                                      uint32_t route_idx,
                                      uint32_t* from_out,
                                      uint32_t* to_out) {
  if (route_idx >= t->tt->n_routes()) {
    return false;
  }
  auto const ridx = nigiri::route_idx_t{route_idx};
  auto const range = t->tt->route_transport_ranges_[ridx];
  *from_out = static_cast<nigiri::transport_idx_t::value_t>(range.from_);
  *to_out = static_cast<nigiri::transport_idx_t::value_t>(range.to_);
  return true;
}

uint16_t nigiri_get_transport_stop_times(nigiri_timetable_t const* t,
                                         uint32_t transport_idx,
                                         uint32_t* stop_locations_out,
                                         int16_t* dep_mams_out,
                                         int16_t* arr_mams_out,
                                         uint16_t max_stops) {
  if (transport_idx >= t->tt->transport_route_.size()) {
    return 0;
  }
  auto const tidx = nigiri::transport_idx_t{transport_idx};
  auto const ridx = t->tt->transport_route_[tidx];
  auto const seq = t->tt->route_location_seq_[ridx];
  auto const n_stops = static_cast<uint16_t>(
      std::min(static_cast<size_t>(max_stops), seq.size()));

  for (uint16_t i = 0; i < n_stops; ++i) {
    if (stop_locations_out) {
      stop_locations_out[i] = static_cast<nigiri::location_idx_t::value_t>(
          nigiri::stop{seq[i]}.location_idx());
    }
    if (dep_mams_out) {
      dep_mams_out[i] =
          (i < n_stops - 1)
              ? t->tt
                    ->event_mam(tidx, nigiri::stop_idx_t{i},
                                nigiri::event_type::kDep)
                    .count()
              : -1;
    }
    if (arr_mams_out) {
      arr_mams_out[i] =
          (i > 0) ? t->tt
                        ->event_mam(tidx, nigiri::stop_idx_t{i},
                                    nigiri::event_type::kArr)
                        .count()
                  : -1;
    }
  }
  return n_stops;
}

uint32_t nigiri_get_transport_display_name(nigiri_timetable_t const* t,
                                           uint32_t transport_idx,
                                           char* buf,
                                           uint32_t buf_len) {
  if (transport_idx >= t->tt->transport_route_.size()) {
    return 0;
  }
  auto const tidx = nigiri::transport_idx_t{transport_idx};
  auto const trip_sections = t->tt->transport_to_trip_section_[tidx];
  if (trip_sections.empty()) {
    return 0;
  }
  auto const merged = t->tt->merged_trips_[trip_sections.front()];
  if (merged.empty()) {
    return 0;
  }
  auto const trip_idx = merged.front();
  auto const name =
      t->tt->get_default_translation(t->tt->trip_display_names_[trip_idx]);
  auto const len = static_cast<uint32_t>(name.length());
  if (buf != nullptr && buf_len > 0) {
    auto const copy_len = std::min(len, buf_len);
    std::memcpy(buf, name.data(), copy_len);
  }
  return len;
}
