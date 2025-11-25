#include "nigiri/loader/netex/load_timetable.h"

#include <charconv>
#include <filesystem>
#include <numeric>
#include <string>

#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "wyhash.h"

#include "pugixml.hpp"

#include "nigiri/loader/get_index.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/loader/register.h"
#include "nigiri/common/sort_by.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

#include "nigiri/loader/netex/intermediate.h"
#include "nigiri/loader/netex/parser.h"
#include "nigiri/loader/netex/stop_places.h"

namespace fs = std::filesystem;

namespace nigiri::loader::netex {

bool is_netex_file(fs::path const& p) {
  return p.extension() == ".xml" || p.extension() == ".XML";
}

cista::hash_t hash(dir const& d) {
  if (d.type() == dir_type::kZip) {
    return d.hash();
  }

  auto h = std::uint64_t{0U};
  auto const hash_file = [&](fs::path const& p) {
    if (!d.exists(p)) {
      h = wyhash64(h, _wyp[0]);
    } else {
      auto const f = d.get_file(p);
      auto const data = f.data();
      h = wyhash(data.data(), data.size(), h, _wyp);
    }
  };

  for (auto const& f : d.list_files("/")) {
    if (is_netex_file(f)) {
      hash_file(f);
    }
  }

  return h;
}

bool applicable(dir const& d) {
  return utl::any_of(d.list_files("/"), is_netex_file);
}

void load_timetable(loader_config const& config,
                    source_idx_t const src,
                    dir const& d,
                    timetable& tt,
                    hash_map<bitfield, bitfield_idx_t>& /*bitfield_indices*/,
                    assistance_times* /*assistance*/,
                    shapes_storage* /*shapes_data*/) {
  auto const global_timer = nigiri::scoped_timer{"netex parser"};
  auto const progress_tracker = utl::get_active_progress_tracker();

  auto const xml_files =
      utl::all(d.list_files(""))  //
      | utl::remove_if([&](fs::path const& f) { return !is_netex_file(f); })  //
      | utl::vec();

  progress_tracker->status("Parse Files")
      .out_bounds(0.F, 90.F)
      .in_high(xml_files.size());
}

}  // namespace nigiri::loader::netex
