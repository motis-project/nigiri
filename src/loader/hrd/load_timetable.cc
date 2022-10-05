#include "nigiri/loader/hrd/load_timetable.h"

#include <execution>

#include "wyhash.h"

#include "utl/helpers/algorithm.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/hrd/service/service_builder.h"
#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/print_transport.h"

namespace nigiri::loader::hrd {

void print_timetable(std::ostream& out, timetable const& tt) {
  auto const reverse = [](std::string s) {
    std::reverse(s.begin(), s.end());
    return s;
  };
  auto const num_days = static_cast<size_t>(
      (tt.date_range_.to_ - tt.date_range_.from_ + 1_days) / 1_days);
  auto ret = std::set<std::string>{};
  for (auto i = 0U; i != tt.transport_stop_times_.size(); ++i) {
    auto const transport_idx = transport_idx_t{i};
    auto const traffic_days =
        tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
    out << "TRAFFIC_DAYS="
        << reverse(
               traffic_days.to_string().substr(traffic_days.size() - num_days))
        << "\n";
    for (auto d = tt.date_range_.from_; d != tt.date_range_.to_;
         d += std::chrono::days{1}) {
      auto const day_idx = day_idx_t{
          static_cast<day_idx_t::value_t>((d - tt.date_range_.from_) / 1_days)};
      if (traffic_days.test(to_idx(day_idx))) {
        date::to_stream(out, "%F", d);
        out << " (day_idx=" << day_idx << ")\n";
        print_transport(tt, out, {transport_idx, day_idx});
      }
    }
  }
}

bool applicable(config const& c, dir const& d) {
  return utl::all_of(
      c.required_files_, [&](std::vector<std::string> const& alt) {
        return alt.empty() || utl::any_of(alt, [&](std::string const& file) {
                 auto const exists = d.exists(c.core_data_ / file);
                 if (!exists) {
                   log(log_lvl::info, "loader.hrd",
                       "missing file for config {}: {}", c.version_.view(),
                       (c.core_data_ / file));
                 }
                 return exists;
               });
      });
}

std::uint64_t hash(config const& c, dir const& d, std::uint64_t const seed) {
  auto h = seed;
  for (auto const& f : stamm::load_files(c, d)) {
    if (!f.has_value()) {
      h = wyhash64(h, _wyp[0]);
    } else {
      auto const data = f.data();
      h = wyhash(data.data(), data.size(), h, _wyp);
    }
  }
  for (auto const& path : d.list_files(c.fplan_)) {
    auto const f = d.get_file(path);
    auto const data = f.data();
    h = wyhash(data.data(), data.size(), h, _wyp);
  }
  return h;
}

void load_timetable(source_idx_t const src,
                    config const& c,
                    dir const& d,
                    timetable& tt) {
  (void)src;
  auto bars = utl::global_progress_bars{false};

  auto st = stamm{c, tt, d};
  service_builder sb{st, tt};

  auto progress_tracker = utl::activate_progress_tracker("nigiri");
  progress_tracker->status("Read Services")
      .in_high(utl::all(d.list_files(c.fplan_))  //
               | utl::transform([&](auto&& f) { return d.file_size(f); })  //
               | utl::sum());
  auto total_bytes_processed = std::uint64_t{0U};

  for (auto const& path : d.list_files(c.fplan_)) {
    log(log_lvl::info, "loader.hrd.services", "loading {}", path);
    auto const file = d.get_file(path);
    sb.add_services(
        c, relative(path, c.fplan_).string().c_str(), file.data(),
        [&](std::size_t const bytes_processed) {
          progress_tracker->update(total_bytes_processed + bytes_processed);
        });
    sb.write_services(src);
    total_bytes_processed += file.data().size();
  }

  tt.location_routes_[location_idx_t{tt.locations_.src_.size() - 1}];

  scoped_timer sort_timer{"sorting trip ids"};

  std::sort(
#if __cpp_lib_execution
      std::execution::par_unseq,
#endif
      begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_),
      [&](pair<trip_id_idx_t, trip_idx_t> const& a,
          pair<trip_id_idx_t, trip_idx_t> const& b) {
        return tt.trip_id_strings_[a.first].view() <
               tt.trip_id_strings_[b.first].view();
      });
  build_footpaths(tt);

  //print_timetable(std::cout, tt);
}

}  // namespace nigiri::loader::hrd
