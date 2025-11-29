#include "nigiri/loader/netex/load_timetable.h"

#include <charconv>
#include <filesystem>
#include <numeric>
#include <string>

#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"
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

#include "utl/parser/arg_parser.h"

namespace fs = std::filesystem;

namespace nigiri::loader::netex {

struct none {};

struct stop {
  std::string_view parent_;
  std::string_view id_;
  std::string_view name_;
  geo::latlng pos_;
};

struct stop_point {
  std::string_view stop_id_;
  bool out_allowed_;
  bool in_allowed_;
};

std::tuple<std::string_view, std::string_view, std::string_view, geo::latlng>
format_as(stop const& x) {
  return {x.parent_, x.id_, x.name_, x.pos_};
}

bool is_xml_file(fs::path const& p) {
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
    if (is_xml_file(f)) {
      hash_file(f);
    }
  }

  return h;
}

bool applicable(dir const& d) {
  return utl::any_of(d.list_files("/"), is_xml_file);
}

std::string_view child_attr(pugi::xml_node n,
                            char const* child,
                            char const* attr) {
  auto const str = n.child(child).attribute(attr).as_string();
  return str == nullptr ? std::string_view{} : std::string_view{str};
}

void load_timetable(loader_config const&,
                    source_idx_t const,
                    dir const& d,
                    timetable&,
                    hash_map<bitfield, bitfield_idx_t>& /*bitfield_indices*/,
                    assistance_times* /*assistance*/,
                    shapes_storage* /*shapes_data*/) {
  auto const global_timer = nigiri::scoped_timer{"netex parser"};

  auto const pt = utl::get_active_progress_tracker();

  auto const xml_files =
      utl::all(d.list_files(""))  //
      | utl::remove_if([&](fs::path const& f) { return !is_xml_file(f); })  //
      | utl::vec();

  pt->status("Parse Files").out_bounds(0.F, 90.F).in_high(xml_files.size());

  utl::parallel_ordered_collect_threadlocal<none>(
      xml_files.size(),
      [&](none&, std::size_t const i) {
        CISTA_UNUSED_PARAM(i)

        auto const f = d.get_file(xml_files.at(i));
        auto const file_content = f.data();
        auto doc = pugi::xml_document{};
        auto const result =
            doc.load_buffer(file_content.data(), file_content.size(),
                            pugi::parse_default | pugi::parse_trim_pcdata);
        utl::verify(result, "Unable to parse XML buffer: {} at offset {}",
                    result.description(), result.offset);

        auto stops = hash_map<std::string_view, stop>{};
        for (auto const s :
             doc.select_nodes("//SiteFrame/stopPlaces/StopPlace | "
                              "//SiteFrame/stopPlaces/Quay")) {
          auto const n = s.node();

          auto const get_global_id = [](pugi::xml_node const x) {
            return x.select_node("keyList/KeyValue/Key[text() = 'GlobalID']")
                .parent()
                .child("Value")
                .child_value();
          };

          auto const get_pos = [](pugi::xml_node const x) {
            return geo::latlng{
                utl::parse<double>(x.select_node("Centroid/Location/Latitude")
                                       .node()
                                       .child_value()),
                utl::parse<double>(x.select_node("Centroid/Location/Longitude")
                                       .node()
                                       .child_value())};
          };

          auto const stop_id = n.attribute("id").as_string();
          auto const global_stop_id = get_global_id(n);
          stops.emplace(stop_id,
                        stop{.parent_ = {},
                             .id_ = global_stop_id ? global_stop_id : stop_id,
                             .name_ = n.child("Name").child_value(),
                             .pos_ = get_pos(n)});

          for (auto const q : n.select_nodes("quays/Quay")) {
            auto const qn = q.node();
            auto const quay_id = qn.attribute("id").as_string();
            auto const global_quay_id = get_global_id(qn);
            stops.emplace(quay_id,
                          stop{.parent_ = stop_id,
                               .id_ = global_quay_id ? global_quay_id : quay_id,
                               .name_ = qn.child("Name").child_value(),
                               .pos_ = get_pos(qn)});
          }
        }

        auto psa = hash_map<std::string_view, stop const*>{};
        for (auto const& a : doc.select_nodes("//ServiceFrame/stopAssignments/"
                                              "PassengerStopAssignment")) {
          auto const n = a.node();
          auto const sstop = child_attr(n, "ScheduledStopPointRef", "ref");
          auto const quay = child_attr(n, "QuayRef", "ref");
          auto const stop_place = child_attr(n, "StopPlaceRef", "ref");
          if (!sstop.empty() && (!quay.empty() || !stop_place.empty())) {
            auto s = end(stops);

            if (!quay.empty()) {
              s = stops.find(quay);
            }
            if (s == end(stops) && !stop_place.empty()) {
              s = stops.find(stop_place);
            }

            if (s != end(stops)) {
              psa[sstop] = &s->second;
            }
          }
        }

        fmt::println("Stops:\n\t{}", fmt::join(stops, "\n\t"));
        for (auto const& [from, to] : psa) {
          fmt::println("{} -> {}", from, format_as(*to));
        }

        auto im = intermediate{};
        return im;
      },
      [](std::size_t const i, intermediate const& im) {
        CISTA_UNUSED_PARAM(i)
        CISTA_UNUSED_PARAM(im)
      },
      pt->update_fn());
}

}  // namespace nigiri::loader::netex
