#include "nigiri/loader/netex/stop_places.h"

#include "utl/helpers/algorithm.h"
#include "utl/parser/arg_parser.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include <cstdlib>
#include <algorithm>
#include <string_view>
#include <utility>

#include "nigiri/logging.h"

#include "nigiri/loader/netex/locale.h"

namespace nigiri::loader::netex {

inline std::pair<double, double> parse_double_pair(
    std::string_view const input) {
  auto* pos = input.data();
  char* end_pos = nullptr;
  auto first = std::strtod(pos, &end_pos);
  utl::verify(pos != end_pos,
              "invalid number pair: {} (failed to parse first number)", input);

  pos = end_pos;
  auto second = std::strtod(pos, &end_pos);
  utl::verify(pos != end_pos,
              "invalid number pair: {} (failed to parse second number)", input);

  return {first, second};
}

geo::latlng parse_location(netex_data& data,
                           netex_ctx const& ctx,
                           pugi::xml_node const& loc_node) {
  if (!loc_node) {
    return {};
  }

  auto crs = ctx.default_crs_;
  if (auto const srs_name = loc_node.attribute("srsName"); srs_name) {
    crs = srs_name.value();
  }

  auto const lon_node = loc_node.child("Longitude");
  auto const lat_node = loc_node.child("Latitude");
  if (lon_node && lat_node) {
    auto const lat = std::clamp(
        utl::parse<double>(loc_node.child_value("Latitude")), -90.0, 90.0);
    auto const lon = std::clamp(
        utl::parse<double>(loc_node.child_value("Longitude")), -180.0, 180.0);
    return crs ? data.proj_transformers_.transform(*crs, lat, lon)
               : geo::latlng{lat, lon};
  }

  if (auto gml_pos =
          loc_node.select_node("*[namespace-uri()='http://www.opengis.net/"
                               "gml/3.2' and local-name()='pos']");
      gml_pos) {
    if (auto const srs_name = gml_pos.node().attribute("srsName"); srs_name) {
      crs = srs_name.value();
    }
    auto const pos_str = gml_pos.node().child_value();
    try {
      auto const [x, y] = parse_double_pair(pos_str);
      return crs ? data.proj_transformers_.transform(*crs, x, y)
                 : geo::latlng{x, y};
    } catch (std::exception const& e) {
      std::cerr << "Error parsing position: " << pos_str
                << ", error: " << e.what() << "\n";
      return {};
    }
  }

  return {};
}

quay parse_quay(netex_data& data,
                netex_ctx const& ctx,
                pugi::xml_node const& quay_node) {
  auto const id = quay_node.attribute("id").value();
  auto const name = quay_node.child_value("Name");
  auto const public_code = quay_node.child_value("PublicCode");
  auto const centroid =
      parse_location(data, ctx, quay_node.child("Centroid").child("Location"));
  auto const locale = ctx.locale_.value_or(netex_locale{});

  auto parent = std::optional<std::string>{};
  if (auto const pzr = quay_node.child("ParentZoneRef")) {
    parent = pzr.attribute("ref").value();
  }

  return {
      .id_ = id,
      .name_ = name,
      .public_code_ = public_code,
      .ssp_public_code_ = "",
      .centroid_ = centroid,
      .parent_ref_ = parent,
      .locale_ = locale,
  };
}

void handle_stop_place(netex_data& data,
                       netex_ctx const& ctx,
                       pugi::xml_node const& spn) {
  auto const centroid =
      parse_location(data, ctx, spn.child("Centroid").child("Location"));
  auto const locale_node = spn.child("Locale");
  auto sp = stop_place{
      .id_ = spn.attribute("id").value(),
      .name_ = spn.child_value("Name"),
      .description_ = spn.child_value("Description"),
      .centroid_ = centroid,
      .locale_ = locale_node ? parse_locale(data, ctx, locale_node)
                             : ctx.locale_.value_or(netex_locale{}),
  };

  for (auto const& xpqn : spn.select_nodes("quays/Quay")) {
    sp.quays_.emplace_back(parse_quay(data, ctx, xpqn.node()));
  }

  auto const& merge_stop_place = [&](stop_place& existing_sp,
                                     stop_place const& new_sp) {
    for (auto const& quay : new_sp.quays_) {
      if (utl::none_of(existing_sp.quays_,
                       [&](auto const& q) { return q.id_ == quay.id_; })) {
        existing_sp.quays_.emplace_back(quay);
      }
    }
  };

  if (auto const psr = spn.child("ParentSiteRef"); psr) {
    sp.parent_ref_ = psr.attribute("ref").value();
  }

  if (auto const existing_it = data.stop_places_.find(sp.id_);
      existing_it != data.stop_places_.end()) {
    auto& existing_sp = existing_it->second;
    merge_stop_place(existing_sp, sp);
  } else {
    auto const id = sp.id_;
    data.stop_places_.emplace(id, std::move(sp));
  }
}

void handle_quay(netex_data& data,
                 netex_ctx const& ctx,
                 pugi::xml_node const& qn) {
  auto quay = parse_quay(data, ctx, qn);
  if (quay.parent_ref_) {
    if (auto it = data.stop_places_.find(*quay.parent_ref_);
        it != data.stop_places_.end()) {
      auto& sp = it->second;
      if (utl::none_of(sp.quays_,
                       [&](auto const& q) { return q.id_ == quay.id_; })) {
        sp.quays_.emplace_back(std::move(quay));
      }
    } else {
      data.quays_with_missing_parents_[quay.id_] = std::move(quay);
    }
  } else {
    data.standalone_quays_[quay.id_] = std::move(quay);
  }
}

void finalize_stop_places(netex_data& data) {
  for (auto& spp : data.stop_places_) {
    auto& sp = spp.second;
    if (sp.parent_ref_) {
      if (auto it = data.stop_places_.find(*sp.parent_ref_);
          it != data.stop_places_.end()) {
        auto& parent_sp = it->second;
        parent_sp.children_.emplace_back(sp.id_);
      } else {
        log(log_lvl::error, "nigiri.loader.netex.stop_places",
            "stop place {} has missing parent {}", sp.id_, *sp.parent_ref_);
      }
    }
  }

  for (auto updated = true;
       updated && !data.quays_with_missing_parents_.empty();) {
    updated = false;
    for (auto it = data.quays_with_missing_parents_.begin();
         it != data.quays_with_missing_parents_.end();) {
      auto& q = it->second;
      if (auto parent_it = data.stop_places_.find(*q.parent_ref_);
          parent_it != data.stop_places_.end()) {
        auto& sp = parent_it->second;
        if (utl::none_of(sp.quays_,
                         [&](auto const& eq) { return eq.id_ == q.id_; })) {
          sp.quays_.emplace_back(q);
        }
        data.quays_with_missing_parents_.erase(it);
        updated = true;
        break;
      } else {
        ++it;
      }
    }
  }

  if (!data.quays_with_missing_parents_.empty()) {
    log(log_lvl::error, "nigiri.loader.netex.stop_places",
        "{} quays have missing parent stop places",
        data.quays_with_missing_parents_.size());
    for (auto const& [id, quay] : data.quays_with_missing_parents_) {
      log(log_lvl::error, "nigiri.loader.netex.stop_places",
          "quay {} has missing parent {}", id, *quay.parent_ref_);
    }
  }
}

}  // namespace nigiri::loader::netex
