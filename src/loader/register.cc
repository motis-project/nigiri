#include "nigiri/loader/register.h"

#include <cassert>

#include "nigiri/loader/gtfs/route.h"
#include "nigiri/loader/gtfs/tz_map.h"

#include "sol/sol.hpp"

#include "nigiri/timetable.h"

#include "utl/get_or_create.h"

namespace nigiri::loader {

gtfs::tz_map dummy_tz_map;

// ==========
// Attributes
// ----------

attribute::attribute(timetable* tt,
                     std::string_view code,
                     translation_idx_t text)
    : code_{code, generic_string::non_owning}, text_{text}, tt_{tt} {}

std::string_view attribute::get_code() const { return code_; }
void attribute::set_code(std::string_view s) { code_.set_owning(s); }

std::string_view attribute::get_text() const {
  return tt_->get_default_translation(text_);
}
translated_str_t attribute::get_text_translations() const {
  return tt_->get(text_);
}
void attribute::set_text(translated_str_t t) {
  text_ = tt_->register_translation(t);
}

// =======
// Agency
// -------

agency::agency(timetable& tt,
               source_idx_t const src,
               std::string_view id,
               translation_idx_t name,
               translation_idx_t url,
               timezone_idx_t const tz_idx,
               gtfs::tz_map& tz_map)
    : src_{src},
      id_{id},
      name_{std::move(name)},
      url_{std::move(url)},
      timezone_idx_{tz_idx},
      tt_{&tt},
      tz_map_{&tz_map} {}

agency::agency(timetable& tt, provider_idx_t const a)
    : id_{tt.strings_.get(tt.providers_[a].id_)},
      name_{tt.providers_[a].name_},
      url_{tt.providers_[a].url_},
      timezone_idx_{tt.providers_[a].tz_},
      tt_{&tt} {}

std::string_view agency::get_id() const { return id_; }

std::string_view agency::get_name() const {
  return tt_->get_default_translation(name_);
}
translated_str_t agency::get_name_translations() const {
  return tt_->get(name_);
}
void agency::set_name(translated_str_t x) {
  name_ = tt_->register_translation(x);
}

std::string_view agency::get_url() const {
  return tt_->get_default_translation(url_);
}
translated_str_t agency::get_url_translations() const { return tt_->get(url_); }
void agency::set_url(translated_str_t x) {
  url_ = tt_->register_translation(x);
}

std::optional<std::string_view> agency::get_timezone() const {
  return gtfs::get_timezone_name(*tt_, timezone_idx_);
}
void agency::set_timezone(std::string_view x) {
  timezone_idx_ = gtfs::get_tz_idx(*tt_, *tz_map_, x);
}

// ========
// Location
// --------

location::location(timetable const& tt, location_idx_t const l)
    : src_{tt.locations_.src_[l]},
      id_{tt.locations_.ids_[l].view()},
      name_{tt.locations_.names_[l]},
      platform_code_{tt.locations_.platform_codes_[l]},
      description_{tt.locations_.descriptions_[l]},
      pos_{tt.locations_.coordinates_[l]},
      type_{tt.locations_.types_[l]},
      parent_{tt.locations_.parents_[l]},
      timezone_idx_{tt.locations_.location_timezones_[l]},
      transfer_time_{tt.locations_.transfer_time_[l]},
      tt_{const_cast<timetable*>(&tt)} {}

location::location(timetable& tt,
                   source_idx_t src,
                   std::string_view id,
                   translation_idx_t name,
                   translation_idx_t platform_code,
                   translation_idx_t desc,
                   geo::latlng pos,
                   location_type type,
                   location_idx_t parent,
                   timezone_idx_t timezone,
                   duration_t transfer_time,
                   gtfs::tz_map& tz_map)
    : src_{src},
      id_{id},
      name_{name},
      platform_code_{platform_code},
      description_{desc},
      pos_{pos},
      type_{type},
      parent_{parent},
      timezone_idx_{timezone},
      transfer_time_{transfer_time},
      tt_{&tt},
      tz_map_{&tz_map} {}

std::string_view location::get_id() const { return id_; }

std::string_view location::get_name() const {
  return tt_->get_default_translation(name_);
}
translated_str_t location::get_name_translations() const {
  return tt_->get(name_);
}
void location::set_name(translated_str_t x) {
  name_ = tt_->register_translation(x);
}

std::string_view location::get_platform_code() const {
  return tt_->get_default_translation(platform_code_);
}
translated_str_t location::get_platform_code_translations() const {
  return tt_->get(platform_code_);
}
void location::set_platform_code(translated_str_t x) {
  platform_code_ = tt_->register_translation(x);
}

std::string_view location::get_description() const {
  return tt_->get_default_translation(description_);
}
translated_str_t location::get_description_translations() const {
  return tt_->get(description_);
}
void location::set_description(translated_str_t x) {
  description_ = tt_->register_translation(x);
}

geo::latlng location::get_pos() const { return pos_; }
void location::set_pos(geo::latlng x) { pos_ = x; }

std::optional<std::string_view> location::get_timezone() const {
  return gtfs::get_timezone_name(*tt_, timezone_idx_);
}
void location::set_timezone(std::string_view tz_name) {
  assert(tz_map_ != nullptr);
  timezone_idx_ = gtfs::get_tz_idx(*tt_, *tz_map_, tz_name);
}

duration_t::rep location::get_transfer_time() const {
  return transfer_time_.count();
}
void location::set_transfer_time(duration_t::rep const x) {
  transfer_time_ = duration_t{x};
}

// =====
// Route
// -----

route::route(timetable& tt,
             source_idx_t const src,
             std::string_view id,
             translation_idx_t short_name,
             translation_idx_t long_name,
             route_type_t const route_type,
             route_color const color,
             provider_idx_t const agency)
    : src_{src},
      id_{id},
      short_name_{short_name},
      long_name_{long_name},
      route_type_{route_type},
      color_{color},
      clasz_{gtfs::to_clasz(to_idx(route_type))},
      agency_{agency},
      tt_{&tt} {}

route::route(timetable& tt, source_idx_t const src, route_id_idx_t const r)
    : src_{src},
      id_{tt.route_ids_[src].ids_.get(r)},
      short_name_{tt.route_ids_[src].route_id_short_names_[r]},
      long_name_{tt.route_ids_[src].route_id_long_names_[r]},
      route_type_{tt.route_ids_[src].route_id_type_[r]},
      color_{tt.route_ids_[src].route_id_colors_[r]},
      agency_{tt.route_ids_[src].route_id_provider_[r]},
      tt_{&tt} {}

std::string_view route::get_id() const { return id_; }

std::string_view route::get_short_name() const {
  return tt_->get_default_translation(short_name_);
}
std::vector<translation> route::get_short_name_translations() const {
  return std::get<std::vector<translation>>(tt_->get(short_name_));
}
void route::set_short_name(translated_str_t x) {
  short_name_ = tt_->register_translation(x);
}

std::string_view route::get_long_name() const {
  return tt_->get_default_translation(long_name_);
}
translated_str_t route::get_long_name_translations() const {
  return tt_->get(long_name_);
}
void route::set_long_name(translated_str_t x) {
  long_name_ = tt_->register_translation(x);
}

route_type_t::value_t route::get_route_type() const {
  return static_cast<route_type_t::value_t>(route_type_);
}
void route::set_route_type(route_type_t::value_t x) {
  route_type_ = route_type_t{x};
}

std::uint32_t route::get_color() const { return to_idx(color_.color_); }
void route::set_color(std::uint32_t const x) { color_.color_ = color_t{x}; }

std::uint32_t route::get_text_color() const {
  return to_idx(color_.text_color_);
}
void route::set_text_color(std::uint32_t const x) {
  color_.text_color_ = color_t{x};
}

clasz route::get_clasz() const { return clasz_; }
void route::set_clasz(clasz const x) { clasz_ = x; }

agency route::get_agency() const { return agency{*tt_, agency_}; }

// ====
// Trip
// ----

trip::trip(timetable& tt,
           source_idx_t src,
           std::string_view id,
           translation_idx_t headsign,
           translation_idx_t short_name,
           translation_idx_t display_name,
           std::string_view vehicle_type_name,
           std::string_view vehicle_type_short_name,
           direction_id_t direction,
           route_id_idx_t route,
           trip_debug dbg)
    : src_{src},
      id_{id},
      headsign_{headsign},
      short_name_{short_name},
      display_name_{display_name},
      vehicle_type_name_{vehicle_type_name},
      vehicle_type_short_name_{vehicle_type_short_name},
      direction_{direction},
      route_{route},
      dbg_{dbg},
      tt_{&tt} {}

std::string_view trip::get_id() const { return id_; }

std::string_view trip::get_headsign() const {
  return tt_->get_default_translation(headsign_);
}
translated_str_t trip::get_headsign_translations() const {
  return tt_->get(headsign_);
}
void trip::set_headsign(translated_str_t x) {
  headsign_ = tt_->register_translation(x);
}

std::string_view trip::get_short_name() const {
  return tt_->get_default_translation(short_name_);
}
translated_str_t trip::get_short_name_translations() const {
  return tt_->get(short_name_);
}
void trip::set_short_name(translated_str_t x) {
  short_name_ = tt_->register_translation(x);
}

std::string_view trip::get_vehicle_type_name() const {
  return vehicle_type_name_;
}

std::string_view trip::get_vehicle_type_short_name() const {
  return vehicle_type_short_name_;
}

std::string_view trip::get_display_name() const {
  return tt_->get_default_translation(display_name_);
}
translated_str_t trip::get_display_name_translations() const {
  return tt_->get(display_name_);
}
void trip::set_display_name(translated_str_t x) {
  display_name_ = tt_->register_translation(x);
}

route trip::get_route() const { return route{*tt_, src_, route_}; }

// ===========
// User Script
// -----------

struct script_runner::impl {
  sol::state lua_;

  sol::protected_function process_attribute_;
  sol::protected_function process_agency_;
  sol::protected_function process_location_;
  sol::protected_function process_route_;
  sol::protected_function process_trip_;
};

script_runner::script_runner() = default;

script_runner::script_runner(std::string const& user_script)
    : impl_{user_script.empty() ? nullptr : std::make_unique<impl>()} {
  if (user_script.empty()) {
    return;
  }

  impl_->lua_.open_libraries(sol::lib::base, sol::lib::string,
                             sol::lib::package);
  impl_->lua_.script(user_script);

  impl_->lua_.new_usertype<geo::latlng>(
      "latlng",  //
      "get_lat", &geo::latlng::lat,  //
      "get_lng", &geo::latlng::lng,  //
      "set_lat", [](geo::latlng& x, double lat) { x.lat_ = lat; },  //
      "set_lng", [](geo::latlng& x, double lng) { x.lng_ = lng; });

  impl_->lua_.new_usertype<translation>(
      "translation",  //
      sol::constructors<translation(std::string_view, std::string_view)>(),  //
      "get_language", &translation::get_language,  //
      "set_language", &translation::set_language,  //
      "get_text", &translation::get_text,  //
      "set_text", &translation::set_text);

  impl_->lua_.new_usertype<attribute>(  //
      "attribute",  //
      "get_code", &attribute::get_code,  //
      "set_code", &attribute::set_code,  //
      "get_text", &attribute::get_text,  //
      "get_text_translations", &attribute::get_text_translations,  //
      "set_text", &attribute::set_text  //
  );

  impl_->lua_.new_usertype<agency>(  //
      "agency",  //
      "get_id", &agency::get_id,  //
      "get_name", &agency::get_name,  //
      "get_name_translations", &agency::get_name_translations,  //
      "set_name", &agency::set_name,  //
      "get_url", &agency::get_url,  //
      "get_url_translations", &agency::get_url_translations,  //
      "set_url", &agency::set_url,  //
      "get_timezone", &agency::get_timezone,  //
      "set_timezone", &agency::set_timezone  //
  );

  impl_->lua_.new_usertype<location>(
      "location",  //
      "get_id", &location::get_id,  //
      "get_name", &location::get_name,  //
      "get_name_translations", &location::get_name_translations,  //
      "set_name", &location::set_name,  //
      "get_platform_code", &location::get_platform_code,  //
      "get_platform_code_translations",
      &location::get_platform_code_translations,  //
      "set_platform_code", &location::set_platform_code,  //
      "get_description", &location::get_description,  //
      "get_description_translations",
      &location::get_description_translations,  //
      "set_description", &location::set_description,  //
      "get_pos", &location::get_pos,  //
      "set_pos", &location::set_pos,  //
      "get_timezone", &location::get_timezone,  //
      "set_timezone", &location::set_timezone,  //
      "get_transfer_time", &location::get_transfer_time,  //
      "set_transfer_time", &location::set_transfer_time  //
  );

  impl_->lua_.new_usertype<route>(  //
      "route",  //
      "get_id", &route::get_id,  //
      "get_short_name", &route::get_short_name,  //
      "get_short_name_translations", &route::get_short_name_translations,  //
      "set_short_name", &route::set_short_name,  //
      "get_long_name", &route::get_long_name,  //
      "get_long_name_translations", &route::get_long_name_translations,  //
      "set_long_name", &route::set_long_name,  //
      "get_route_type", &route::get_route_type,  //
      "set_route_type", &route::set_route_type,  //
      "get_color", &route::get_color,  //
      "set_color", &route::set_color,  //
      "get_clasz", &route::get_clasz,  //
      "set_clasz", &route::set_clasz,  //
      "get_text_color", &route::get_text_color,  //
      "set_text_color", &route::set_text_color,  //
      "get_agency", &route::get_agency  //
  );

  impl_->lua_.new_usertype<trip>(  //
      "trip",  //
      "get_id", &trip::get_id,  //
      "get_headsign", &trip::get_headsign,  //
      "get_headsign_translations", &trip::get_headsign_translations,  //
      "set_headsign", &trip::set_headsign,  //
      "get_short_name", &trip::get_short_name,  //
      "get_short_name_translations", &trip::get_short_name_translations,  //
      "set_short_name", &trip::set_short_name,  //
      "get_vehicle_type_name", &trip::get_vehicle_type_name,  //
      "get_vehicle_type_short_name", &trip::get_vehicle_type_short_name,  //
      "get_display_name", &trip::get_display_name,  //
      "get_display_name_translations",
      &trip::get_display_name_translations,  //
      "set_display_name", &trip::set_display_name,  //
      "get_route", &trip::get_route  //
  );

  impl_->process_attribute_ = impl_->lua_["process_attribute"];
  impl_->process_agency_ = impl_->lua_["process_agency"];
  impl_->process_location_ = impl_->lua_["process_location"];
  impl_->process_route_ = impl_->lua_["process_route"];
  impl_->process_trip_ = impl_->lua_["process_trip"];

  log(log_lvl::info, "nigiri.loader.user_script",
      "user script handlers: "
      "attribute={}, agency={}, location={}, route={}, trip={}",
      impl_->process_attribute_.valid(), impl_->process_agency_.valid(),
      impl_->process_location_.valid(), impl_->process_route_.valid(),
      impl_->process_trip_.valid());
}

script_runner::~script_runner() = default;

template <typename T>
bool process(sol::protected_function const& process, T& t) {
  if (process.valid()) {
    auto result = process(t);
    if (!result.valid()) {
      sol::error err = result;
      log(log_lvl::error, "nigiri.loader.user_script",
          "user script failed: type={}, error={}", cista::type_str<T>(),
          err.what());
      return true;
    }
    if (result.get_type() == sol::type::boolean) {
      return result.template get<bool>();
    }
  }
  return true;
}

bool process_attribute(script_runner const& r, attribute& x) {
  if (r.impl_ == nullptr) {
    return true;
  }
  return process(r.impl_->process_attribute_, x);
}

bool process_location(script_runner const& r, location& x) {
  if (r.impl_ == nullptr) {
    return true;
  }
  return process(r.impl_->process_location_, x);
}

bool process_agency(script_runner const& r, agency& x) {
  if (r.impl_ == nullptr) {
    return true;
  }
  return process(r.impl_->process_agency_, x);
}

bool process_route(script_runner const& r, route& x) {
  if (r.impl_ == nullptr) {
    return true;
  }
  return process(r.impl_->process_route_, x);
}

bool process_trip(script_runner const& r, trip& x) {
  if (r.impl_ == nullptr) {
    return true;
  }
  return process(r.impl_->process_trip_, x);
}

attribute_idx_t register_attribute(timetable& tt, attribute const& a) {
  auto const idx = tt.attributes_.size();
  tt.attributes_.push_back(
      nigiri::attribute{.code_ = a.get_code(), .text_ = a.text_});
  return attribute_idx_t{idx};
}

provider_idx_t register_agency(timetable& tt, agency const& a) {
  auto const idx = tt.providers_.size();
  tt.providers_.emplace_back(provider{tt.strings_.store(a.id_), a.name_, a.url_,
                                      a.timezone_idx_, a.src_});
  tt.provider_id_to_idx_.emplace_back(idx);
  return provider_idx_t{idx};
}

location_idx_t register_location(timetable& tt, location const& l) {
  auto& loc = tt.locations_;

  auto const next_idx = static_cast<location_idx_t::value_t>(loc.names_.size());
  auto const l_idx = location_idx_t{next_idx};
  auto const [it, is_new] = loc.location_id_to_idx_.emplace(
      owning_location_id{.id_ = l.id_, .src_ = l.src_}, l_idx);

  if (is_new) {
    utl::verify(next_idx <= footpath::kMaxTarget, "MAX={} locations reached",
                footpath::kMaxTarget);

    loc.names_.emplace_back(l.name_);
    loc.platform_codes_.emplace_back(l.platform_code_);
    loc.descriptions_.emplace_back(l.description_);
    loc.coordinates_.emplace_back(l.pos_);
    loc.ids_.emplace_back(l.id_);
    loc.src_.emplace_back(l.src_);
    loc.types_.emplace_back(l.type_);
    loc.location_timezones_.emplace_back(l.timezone_idx_);
    loc.equivalences_.emplace_back();
    loc.children_.emplace_back();
    loc.preprocessing_footpaths_out_.emplace_back();
    loc.preprocessing_footpaths_in_.emplace_back();
    loc.transfer_time_.emplace_back(l.transfer_time_);
    loc.parents_.emplace_back(l.parent_);
  } else {
    assert(false && "duplicate station");
    log(log_lvl::error, "timetable.register_location", "duplicate station {}",
        l.id_);
  }

  assert(loc.names_.size() == next_idx + 1);
  assert(loc.platform_codes_.size() == next_idx + 1);
  assert(loc.descriptions_.size() == next_idx + 1);
  assert(loc.coordinates_.size() == next_idx + 1);
  assert(loc.ids_.size() == next_idx + 1);
  assert(loc.src_.size() == next_idx + 1);
  assert(loc.types_.size() == next_idx + 1);
  assert(loc.location_timezones_.size() == next_idx + 1);
  assert(loc.equivalences_.size() == next_idx + 1);
  assert(loc.children_.size() == next_idx + 1);
  assert(loc.preprocessing_footpaths_out_.size() == next_idx + 1);
  assert(loc.preprocessing_footpaths_in_.size() == next_idx + 1);
  assert(loc.transfer_time_.size() == next_idx + 1);
  assert(loc.parents_.size() == next_idx + 1);

  return it->second;
}

route_id_idx_t register_route(timetable& tt, route const& r) {
  auto& route_id = tt.route_ids_[r.src_];
  auto const idx = route_id.ids_.store(r.id_);
  route_id.route_id_short_names_.emplace_back(r.short_name_);
  route_id.route_id_long_names_.emplace_back(r.long_name_);
  route_id.route_id_colors_.emplace_back(r.color_);
  route_id.route_id_type_.emplace_back(r.route_type_);
  route_id.route_id_provider_.emplace_back(r.agency_);
  route_id.route_id_trips_.emplace_back(std::initializer_list<trip_idx_t>{});
  return idx;
}

trip_idx_t register_trip(timetable& tt, trip const& t) {
  auto const trip_idx = trip_idx_t{tt.trip_ids_.size()};

  auto const trip_id_idx = trip_id_idx_t{tt.trip_id_strings_.size()};

  if (t.route_ != route_id_idx_t::invalid()) {  // HRD
    tt.route_ids_[t.src_].route_id_trips_[t.route_].push_back(trip_idx);
    tt.trip_direction_id_.resize(tt.n_trips() + to_idx(trip_id_idx) + 1U);
    tt.trip_direction_id_.set(trip_idx, t.direction_ == direction_id_t{1U});
  }
  tt.trip_route_id_.emplace_back(t.route_);

  tt.trip_id_strings_.emplace_back(t.id_);
  tt.trip_id_src_.emplace_back(t.src_);

  tt.trip_id_to_idx_.emplace_back(trip_id_idx, trip_idx);
  tt.trip_short_names_.emplace_back(t.short_name_);
  tt.trip_display_names_.emplace_back(t.display_name_);
  tt.trip_ids_.emplace_back().emplace_back(trip_id_idx);
  tt.trip_debug_.emplace_back().emplace_back(t.dbg_);
  tt.trip_transport_ranges_.emplace_back_empty();

  return trip_idx;
}

}  // namespace nigiri::loader
