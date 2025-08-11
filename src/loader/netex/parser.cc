#include "nigiri/loader/netex/stop_places.h"

#include <string_view>

#include "nigiri/loader/netex/locale.h"
#include "nigiri/loader/netex/stop_places.h"

namespace nigiri::loader::netex {

netex_ctx get_frame_ctx(netex_data& data,
                        netex_ctx const& parent_ctx,
                        pugi::xml_node const& frame) {
  auto ctx = parent_ctx;

  if (auto const frame_defaults = frame.child("FrameDefaults")) {
    if (auto const default_locale = frame_defaults.child("DefaultLocale")) {
      ctx.locale_ = parse_locale(data, parent_ctx, default_locale);
    }
    if (auto const default_crs =
            frame_defaults.child("DefaultLocationSystem")) {
      ctx.default_crs_ = default_crs.child_value();
    }
  }

  return ctx;
}

void parse_general_frame(netex_data& data,
                         netex_ctx const& ctx,
                         pugi::xml_node const& gen_frame) {
  for (auto const& xpn : gen_frame.select_nodes("members/StopPlace")) {
    handle_stop_place(data, ctx, xpn.node());
  }
  for (auto const& xpn : gen_frame.select_nodes("members/Quay")) {
    handle_quay(data, ctx, xpn.node());
  }
}

void parse_site_frame(netex_data& data,
                      netex_ctx const& ctx,
                      pugi::xml_node const& gen_frame) {
  for (auto const& xpn : gen_frame.select_nodes("stopPlaces/StopPlace")) {
    handle_stop_place(data, ctx, xpn.node());
  }
}

void parse_composite_frame(netex_data& data,
                           netex_ctx const& ctx,
                           pugi::xml_node const& comp_frame) {
  for (auto const frame : comp_frame.child("frames").children()) {
    auto child_ctx = get_frame_ctx(data, ctx, frame);
    switch (cista::hash(std::string_view{frame.name()})) {
      case cista::hash("SiteFrame"):
        parse_site_frame(data, child_ctx, frame);
        break;
      case cista::hash("GeneralFrame"):
        parse_general_frame(data, child_ctx, frame);
        break;
      default: break;
    }
  }
}

void parse_netex_file(netex_data& data,
                      loader_config const& config,
                      pugi::xml_document const& doc) {
  auto default_locale = get_default_locale(data, config);
  auto default_ctx = netex_ctx{
      .locale_ = default_locale,
  };

  auto const root = doc.document_element();
  for (auto const frame : root.child("dataObjects").children()) {
    auto ctx = get_frame_ctx(data, default_ctx, frame);
    switch (cista::hash(std::string_view{frame.name()})) {
      case cista::hash("CompositeFrame"):
        parse_composite_frame(data, ctx, frame);
        break;
      case cista::hash("SiteFrame"): parse_site_frame(data, ctx, frame); break;
      case cista::hash("GeneralFrame"):
        parse_general_frame(data, ctx, frame);
        break;
      default: break;
    }
  }
}

}  // namespace nigiri::loader::netex
