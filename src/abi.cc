#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

#include "date/date.h"

#include "utl/helpers/algorithm.h"
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
#include "nigiri/timetable.h"

#include "nigiri/common/interval.h"
#include "cista/memory_holder.h"

using namespace date;

struct nigiri_timetable {
  std::shared_ptr<nigiri::timetable> tt;
  std::shared_ptr<nigiri::rt_timetable> rtt;
};

nigiri_timetable_t* nigiri_load_from_dir(nigiri::loader::dir const& d,
                                         int64_t from_ts,
                                         int64_t to_ts) {
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

  t->tt->date_range_ = {floor<days>(std::chrono::system_clock::from_time_t(
                            static_cast<time_t>(from_ts))),
                        floor<days>(std::chrono::system_clock::from_time_t(
                            static_cast<time_t>(to_ts)))};

  nigiri::loader::register_special_stations(*t->tt);
  (*c)->load({}, src, d, *t->tt);
  nigiri::loader::finalize(*t->tt);

  t->rtt = std::make_shared<nigiri::rt_timetable>(
      nigiri::rt::create_rt_timetable(*t->tt, t->tt->date_range_.from_));
  return t;
}

nigiri_timetable_t* nigiri_load(const char* path,
                                int64_t from_ts,
                                int64_t to_ts) {
  auto const progress_tracker = utl::activate_progress_tracker("libnigiri");
  auto const silencer = utl::global_progress_bars{true};

  auto const tt_path = std::filesystem::path{path};
  auto const d = nigiri::loader::make_dir(tt_path);
  return nigiri_load_from_dir(*d, from_ts, to_ts);
}

void nigiri_destroy(const nigiri_timetable_t* t) { delete t; }

int64_t nigiri_get_start_day_ts(const nigiri_timetable_t* t) {
  return std::chrono::system_clock::to_time_t(
      t->tt->internal_interval_days().from_);
}

uint16_t nigiri_get_day_count(const nigiri_timetable_t* t) {
  return static_cast<uint16_t>(t->tt->internal_interval_days().size().count());
}

uint32_t nigiri_get_transport_count(const nigiri_timetable_t* t) {
  return t->tt->transport_route_.size();
}

nigiri_transport_t* nigiri_get_transport(const nigiri_timetable_t* t,
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

void nigiri_destroy_transport(const nigiri_transport_t* transport) {
  delete[] transport->event_mams;
  delete transport;
}

bool nigiri_is_transport_active(const nigiri_timetable_t* t,
                                const uint32_t transport_idx,
                                uint16_t day_idx) {
  auto const tidx = nigiri::transport_idx_t{transport_idx};
  return t->tt->bitfields_[t->tt->transport_traffic_days_[tidx]].test(day_idx);
}

nigiri_route_t* nigiri_get_route(const nigiri_timetable_t* t, uint32_t idx) {
  auto const ridx = nigiri::route_idx_t{idx};
  auto stops = t->tt->route_location_seq_[ridx];
  auto const n_stops = stops.size();
  auto route_stops = new nigiri_route_stop_t[n_stops];
  std::memcpy(route_stops, &stops.front(),
              sizeof(nigiri_route_stop_t) * n_stops);

  auto route = new nigiri_route_t;

  route->stops = route_stops;
  route->n_stops = static_cast<uint16_t>(n_stops);
  route->clasz =
      static_cast<uint16_t>(t->tt->route_section_clasz_[ridx].front());
  return route;
}

void nigiri_destroy_route(const nigiri_route_t* route) {
  delete[] route->stops;
  delete route;
}

uint32_t nigiri_get_location_count(const nigiri_timetable_t* t) {
  return t->tt->n_locations();
}

nigiri_location_t* nigiri_get_location(const nigiri_timetable_t* t,
                                       uint32_t idx) {
  auto const lidx = nigiri::location_idx_t{idx};
  auto location = new nigiri_location_t;
  auto l = t->tt->locations_.get(lidx);
  location->name = l.name_.data();
  location->name_len = static_cast<uint32_t>(l.name_.length());
  location->id = l.id_.data();
  location->id_len = static_cast<uint32_t>(l.id_.length());
  location->lat = l.pos_.lat_;
  location->lon = l.pos_.lng_;
  location->transfer_time = static_cast<uint16_t>(l.transfer_time_.count());
  location->parent =
      l.parent_ == nigiri::location_idx_t::invalid()
          ? 0
          : static_cast<nigiri::location_idx_t::value_t>(l.parent_);
  return location;
}

void nigiri_destroy_location(const nigiri_location_t* location) {
  delete location;
}

void nigiri_update_with_rt_from_buf(const nigiri_timetable_t* t,
                                    std::string_view protobuf,
                                    void (*callback)(nigiri_event_change_t,
                                                     void* context),
                                    void* context) {
  auto const src = nigiri::source_idx_t{0U};
  auto const tag = "";

  auto const rtt_callback =
      [&](nigiri::transport const transport, nigiri::stop_idx_t const stop_idx,
          nigiri::event_type const ev_type, nigiri::duration_t const delay,
          bool const cancelled) {
        nigiri_event_change_t const c = {
            .transport_idx =
                static_cast<nigiri::transport_idx_t::value_t>(transport.t_idx_),
            .day_idx = static_cast<nigiri::day_idx_t::value_t>(transport.day_),
            .stop_idx = stop_idx,
            .is_departure = ev_type != nigiri::event_type::kArr,
            .delay = delay.count(),
            .cancelled = cancelled};
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

void nigiri_update_with_rt(const nigiri_timetable_t* t,
                           const char* gtfsrt_pb_path,
                           void (*callback)(nigiri_event_change_t,
                                            void* context),
                           void* context) {
  auto const file = cista::mmap{gtfsrt_pb_path, cista::mmap::protection::READ};
  return nigiri_update_with_rt_from_buf(t, file.view(), callback, context);
}