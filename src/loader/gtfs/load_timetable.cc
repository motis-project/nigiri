#include "nigiri/loader/gtfs/load_timetable.h"

#include <filesystem>
#include <numeric>
#include <string>

#include "boost/algorithm/string.hpp"

#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/parallel_for.h"
#include "utl/parser/cstr.h"
#include "utl/pipes/accumulate.h"
#include "utl/pipes/all.h"
#include "utl/pipes/remove_if.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/services.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/stop_time.h"
#include "nigiri/loader/gtfs/transfer.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace fs = std::filesystem;
using std::get;

namespace nigiri::loader::gtfs {

constexpr auto const required_files = {kAgencyFile, kStopFile, kRoutesFile,
                                       kTripsFile, kStopTimesFile};

cista::hash_t hash(fs::path const& path) {
  auto hash = cista::BASE_HASH;
  auto const hash_file = [&](fs::path const& p) {
    if (!fs::is_regular_file(p)) {
      return;
    }
    cista::mmap m{p.generic_string().c_str(), cista::mmap::protection::READ};
    hash = cista::hash_combine(
        cista::hash(std::string_view{
            reinterpret_cast<char const*>(m.begin()),
            std::min(static_cast<size_t>(50 * 1024 * 1024), m.size())}),
        hash);
  };

  for (auto const& file_name : required_files) {
    hash_file(path / file_name);
  }
  hash_file(path / kCalenderFile);
  hash_file(path / kCalendarDatesFile);

  return hash;
}

bool applicable(dir const& d) {
  for (auto const& file_name : required_files) {
    if (!d.exists(file_name)) {
      return false;
    }
  }
  return d.exists(kCalenderFile) || d.exists(kCalendarDatesFile);
}

void fix_flixtrain_transfers(trip_map& trips,
                             hash_map<stop_pair, transfer>& transfers) {
  for (auto const& id_prefix : {"FLIXBUS:FLX", "FLIXBUS:K"}) {
    for (auto const& t : trips) {
      if (!boost::starts_with(t.first, id_prefix)) {
        continue;
      }
      for (auto const [dep_entry, arr_entry] :
           utl::pairwise(t.second->stop_times_)) {
        auto& dep = dep_entry.second;
        auto& arr = arr_entry.second;

        if (dep.stop_ == nullptr) {
          continue;  // already gone
        }

        auto const& dep_name = dep.stop_->name_;
        auto const& arr_name = arr.stop_->name_;
        if (utl::get_until(utl::cstr{dep_name}, ',') !=
            utl::get_until(utl::cstr{arr_name}, ',')) {
          continue;  // different towns
        }

        // normal case: bus stop after train stop
        auto const arr_duplicate =
            static_cast<bool>(boost::ifind_first(dep_name, "train")) &&
            !static_cast<bool>(boost::ifind_first(arr_name, "train")) &&
            dep.dep_.time_ == arr.arr_.time_ &&
            arr.arr_.time_ == arr.dep_.time_;

        // may happen on last stop: train stop after bus_stop
        auto const dep_duplicate =
            !static_cast<bool>(boost::ifind_first(dep_name, "train")) &&
            static_cast<bool>(boost::ifind_first(arr_name, "train")) &&
            dep.dep_.time_ == arr.arr_.time_ &&
            dep.arr_.time_ == dep.dep_.time_;

        if (arr_duplicate || dep_duplicate) {
          constexpr const auto kWalkSpeed = 1.5;
          auto dur = static_cast<int>(
              geo::distance(dep.stop_->coord_, arr.stop_->coord_) / kWalkSpeed /
              60);
          transfers.insert({{dep.stop_, arr.stop_},
                            transfer{.type_ = transfer::type::kGenerated,
                                     .minutes_ = u8_minutes{dur}}});
          transfers.insert({{arr.stop_, dep.stop_},
                            {.type_ = transfer::type::kGenerated,
                             .minutes_ = u8_minutes{dur}}});
        }

        if (arr_duplicate) {
          arr.stop_ = nullptr;
        }
        if (dep_duplicate) {
          dep.stop_ = nullptr;
        }
      }

      utl::erase_if(t.second->stop_times_,
                    [](auto const& s) { return s.second.stop_ == nullptr; });
    }
  }
}

void load_timetable(source_idx_t src, dir const& d, timetable& tt) {
  nigiri::scoped_timer const global_timer{"gtfs parser"};

  auto const load = [&](std::string_view file_name) -> file {
    return d.exists(file_name) ? d.get_file(file_name) : file{};
  };

  auto const agencies = read_agencies(tt, load(kAgencyFile).data());
  auto const stops = read_stops(load(kStopTimesFile).data());
  auto const routes = read_routes(agencies, load(kRoutesFile).data());
  auto const calendar = read_calendar(load(kCalenderFile).data());
  auto const dates = read_calendar_date(load(kCalendarDatesFile).data());
  auto const traffic_days = merge_traffic_days(calendar, dates);
  auto transfers = read_transfers(stops, load(kTransfersFile).data());
  auto [trips, blocks] =
      read_trips(routes, traffic_days, load(kTripsFile).data());
  read_frequencies(trips, load(kFrequenciesFile).data());
  read_stop_times(trips, stops, load(kStopTimesFile).data());

  fix_flixtrain_transfers(trips, transfers);

  for (auto& [_, trip] : trips) {
    trip->interpolate();
  }

  // TODO(felix) put data into timetable
}

}  // namespace nigiri::loader::gtfs
