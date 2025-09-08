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
                    hash_map<bitfield, bitfield_idx_t>& /*bitfield_indices*/,
                    assistance_times* /*assistance*/,
                    shapes_storage* /*shapes_data*/) {
  auto const global_timer = nigiri::scoped_timer{"netex parser"};
  auto const progress_tracker = utl::get_active_progress_tracker();

  auto const xml_files = utl::all(d.list_files(""))  //
                         | utl::remove_if([&](fs::path const& f) {
                             return !is_netex_file(d, f);
                           })  //
                         | utl::vec();

  progress_tracker->status("Parse Files")
      .out_bounds(0.F, 90.F)
      .in_high(xml_files.size());

  auto data = netex_data{.tt_ = tt,
                         .script_runner_ = script_runner{config.user_script_}};

  for (auto const& fp : xml_files) {
    auto const f = d.get_file(fp);
    auto const file_content = f.data();
    auto doc = pugi::xml_document{};
    auto const result =
        doc.load_buffer(file_content.data(), file_content.size(),
                        pugi::parse_default | pugi::parse_trim_pcdata);
    utl::verify(result, "Unable to parse XML buffer: {} at offset {}",
                result.description(), result.offset);

    parse_netex_file(data, config, doc);

    progress_tracker->increment();
  }

  progress_tracker->status("Building Timetable")
      .out_bounds(90.F, 100.F)
      .in_high(1);

  finalize_stop_places(data);

  auto lang_map = hash_map<std::string, language_idx_t>{};
  auto const get_lang_idx = [&](std::string const& lang) {
    return utl::get_or_create(lang_map, lang, [&]() {
      auto const lang_idx = language_idx_t{tt.languages_.size()};
      tt.languages_.emplace_back(lang);
      return lang_idx;
    });
  };

  for (auto& [_, sp] : data.stop_places_) {
    auto loc = location{sp.id_,
                        sp.name_,
                        "",
                        sp.description_,
                        sp.centroid_,
                        src,
                        location_type::kStation,
                        location_idx_t::invalid(),
                        sp.locale_.tz_idx_,
                        2_minutes,
                        {},
                        tt,
                        data.timezones_};
    if (!process_location(data.script_runner_, loc)) {
      continue;
    }

    sp.location_idx_ = register_location(tt, loc);

    if (!sp.alt_names_.empty()) {
      auto anb = tt.locations_.alt_names_[sp.location_idx_];
      for (auto const& an : sp.alt_names_) {
        auto const an_idx =
            alt_name_idx_t{tt.locations_.alt_name_strings_.size()};
        tt.locations_.alt_name_strings_.emplace_back(an.name_);
        tt.locations_.alt_name_langs_.emplace_back(get_lang_idx(an.language_));
        anb.push_back(an_idx);
      }
    }

    for (auto& q : sp.quays_) {
      auto quay_loc = location{q.id_,
                               q.name_,
                               q.public_code_,
                               "",
                               q.centroid_,
                               src,
                               location_type::kTrack,
                               sp.location_idx_,
                               q.locale_.tz_idx_,
                               2_minutes,
                               {},
                               tt,
                               data.timezones_};
      if (!process_location(data.script_runner_, quay_loc)) {
        continue;
      }

      q.location_idx_ = register_location(tt, quay_loc);

      tt.locations_.parents_[q.location_idx_] = sp.location_idx_;
      tt.locations_.children_[sp.location_idx_].emplace_back(q.location_idx_);
    }
  }

  for (auto const& [_, sp] : data.stop_places_) {
    if (sp.parent_ref_) {
      if (auto it = data.stop_places_.find(*sp.parent_ref_);
          it != data.stop_places_.end()) {
        auto& parent_sp = it->second;
        if (sp.location_idx_ == location_idx_t::invalid() ||
            parent_sp.location_idx_ == location_idx_t::invalid()) {
          continue;
        }
        tt.locations_.parents_[sp.location_idx_] = parent_sp.location_idx_;
        tt.locations_.children_[parent_sp.location_idx_].emplace_back(
            sp.location_idx_);
      }
    }
  }

  // Add standalone quays as stations
  for (auto& [_, q] : data.standalone_quays_) {
    auto loc = location{q.id_,
                        q.name_,
                        q.public_code_,
                        "",
                        q.centroid_,
                        src,
                        location_type::kStation,
                        location_idx_t::invalid(),
                        q.locale_.tz_idx_,
                        2_minutes,
                        {},
                        tt,
                        data.timezones_};
    if (process_location(data.script_runner_, loc)) {
      q.location_idx_ = register_location(tt, loc);
    }
  }

  progress_tracker->increment();
}

}  // namespace nigiri::loader::netex
