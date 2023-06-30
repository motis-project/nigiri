#include "nigiri/loader/hrd/load_timetable.h"

#include <execution>
#include <filesystem>

#include "wyhash.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/hrd/service/service_builder.h"
#include "nigiri/loader/hrd/stamm/stamm.h"

namespace fs = std::filesystem;

namespace nigiri::loader::hrd {

bool applicable(config const& c, dir const& d) {
  return utl::all_of(
      c.required_files_, [&](std::vector<std::string> const& alt) {
        return alt.empty() || utl::any_of(alt, [&](std::string const& file) {
                 auto const path =
                     (c.prefix(d) / c.core_data_ / file).lexically_normal();
                 auto const exists = d.exists(path);
                 if (!exists) {
                   log(log_lvl::info, "loader.hrd",
                       "input={}, missing file for config {}: {}", d.path(),
                       c.version_.view(), path);
                 }
                 return exists;
               });
      });
}

std::uint64_t hash(config const& c, dir const& d, std::uint64_t const seed) {
  if (d.type() == dir_type::kZip) {
    return d.hash();
  }

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
  auto st = stamm{c, tt, d};

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Read Services")
      .in_high(utl::all(d.list_files(c.prefix(d) / c.fplan_))  //
               | utl::remove_if([&](fs::path const& f) {
                   return !c.fplan_file_extension_.empty() &&
                          f.extension() != c.fplan_file_extension_;
                 })  //
               | utl::transform(
                     [&](fs::path const& f) { return d.file_size(f); })  //
               | utl::sum());
  auto total_bytes_processed = std::uint64_t{0U};

  auto sb = service_builder{st, tt};
  for (auto const& path : d.list_files(c.prefix(d) / c.fplan_)) {
    if (path.filename().generic_string().starts_with(".") ||
        (!c.fplan_file_extension_.empty() &&
         path.extension() != c.fplan_file_extension_)) {
      continue;
    }

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

  sb.write_location_routes();
}

}  // namespace nigiri::loader::hrd
