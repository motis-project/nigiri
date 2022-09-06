#include "nigiri/loader/hrd/load_timetable.h"

#include <execution>

#include "utl/helpers/algorithm.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/hrd/service/service_builder.h"
#include "nigiri/loader/hrd/stamm/stamm.h"

namespace nigiri::loader::hrd {

bool applicable(config const& c, dir const& d) {
  return utl::all_of(
      c.required_files_, [&](std::vector<std::string> const& alt) {
        return alt.empty() || utl::any_of(alt, [&](std::string const& file) {
                 auto const exists = d.exists(c.core_data_ / file);
                 if (!exists) {
                   std::clog << "missing file for config " << c.version_.view()
                             << ": " << (c.core_data_ / file) << "\n";
                 }
                 return exists;
               });
      });
}

void load_timetable(source_idx_t const src,
                    config const& c,
                    dir const& d,
                    timetable& tt) {
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
        c, relative(path, c.fplan_).c_str(), file.data(),
        [&](std::size_t const bytes_processed) {
          progress_tracker->update(total_bytes_processed + bytes_processed);
        });
    sb.write_services(src);
    total_bytes_processed += file.data().size();
  }

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
}

}  // namespace nigiri::loader::hrd