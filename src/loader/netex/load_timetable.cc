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
#include "nigiri/common/sort_by.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

#include "nigiri/loader/netex/parser.h"
#include "nigiri/loader/netex/stop_places.h"

namespace fs = std::filesystem;

namespace nigiri::loader::netex {

bool is_netex_file(dir const& /*d*/, fs::path const& p) {
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
    if (is_netex_file(d, f)) {
      hash_file(f);
    }
  }

  return h;
}

bool applicable(dir const& d) {
  return utl::any_of(d.list_files("/"),
                     [&](fs::path const& p) { return is_netex_file(d, p); });
}

void load_timetable(loader_config const& config,
                    source_idx_t const src,
                    dir const& d,
                    timetable& tt,
                    assistance_times* assistance,
                    shapes_storage* shapes_data) {
  auto local_bitfield_indices = hash_map<bitfield, bitfield_idx_t>{};
  load_timetable(config, src, d, tt, local_bitfield_indices, assistance,
                 shapes_data);
}

void load_timetable(loader_config const& config,
                    source_idx_t const src,
                    dir const& d,
                    timetable& tt,
                    hash_map<bitfield, bitfield_idx_t>& bitfield_indices,
                    assistance_times* assistance,
                    shapes_storage* shapes_data) {
  auto const global_timer = nigiri::scoped_timer{"netex parser"};
  auto const progress_tracker = utl::get_active_progress_tracker();

  auto const xml_files = utl::all(d.list_files("/"))  //
                         | utl::remove_if([&](fs::path const& f) {
                             return !is_netex_file(d, f);
                           })  //
                         | utl::vec();

  progress_tracker->status("Parse Files")
      .out_bounds(0.F, 90.F)
      .in_high(xml_files.size());

  auto data = netex_data{.tt_ = tt};

  for (auto const& fp : xml_files) {
    auto const f = d.get_file(fp);
    auto const file_content = f.data();
    auto doc = pugi::xml_document{};
    auto const result =
        doc.load_buffer(file_content.data(), file_content.size(),
                        pugi::parse_default | pugi::parse_trim_pcdata);
    utl::verify(result, "Unable to parse XML buffer: {} at offset {}",
                result.description(), result.offset);

    parse_netex_file(data, config, src, tt, doc);

    progress_tracker->increment();
  }

  progress_tracker->status("Building Timetable")
      .out_bounds(90.F, 100.F)
      .in_high(1);

  finalize_stop_places(data);

  auto empty_idx_vec = vector<location_idx_t>{};
  for (auto& spp : data.stop_places_) {
    auto& sp = spp.second;

    sp.location_idx_ = tt.locations_.register_location(
        location{sp.id_, sp.name_, "", sp.description_, sp.centroid_, src,
                 location_type::kStation, location_idx_t::invalid(),
                 sp.locale_.tz_idx_, 2_minutes, it_range{empty_idx_vec}});

    for (auto& q : sp.quays_) {
      q.location_idx_ = tt.locations_.register_location(
          location{q.id_, q.name_, q.public_code_, "", q.centroid_, src,
                   location_type::kTrack, sp.location_idx_, q.locale_.tz_idx_,
                   2_minutes, it_range{empty_idx_vec}});
    }

    // TODO: children
  }

  // Add standalone quays as stations
  for (auto& qp : data.standalone_quays_) {
    auto& q = qp.second;

    q.location_idx_ = tt.locations_.register_location(
        location{q.id_, q.name_, q.public_code_, "", q.centroid_, src,
                 location_type::kStation, location_idx_t::invalid(),
                 q.locale_.tz_idx_, 2_minutes, it_range{empty_idx_vec}});
  }

  progress_tracker->increment();
}

}  // namespace nigiri::loader::netex
