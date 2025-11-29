#pragma once

namespace nigiri::loader::netex {

using str_idx_t = cista::strong<std::uint32_t, struct _str_idx>;

template <typename T>
using ref_t = std::variant<T, str_idx_t>;

enum class transport_mode {
  all,
  unknown,
  bus,
  trolleyBus,
  tram,
  coach,
  rail,
  intercityRail,
  urbanRail,
  metro,
  air,
  water,
  cableway,
  funicular,
  snowAndIce,
  taxi,
  selfDrive
};

transport_mode parse_transport_mode(std::string_view s) {
  switch (cista::hash(s)) {
    case cista::hash("all"): return transport_mode::all;
    case cista::hash("unknown"): return transport_mode::unknown;
    case cista::hash("bus"): return transport_mode::bus;
    case cista::hash("trolleyBus"): return transport_mode::trolleyBus;
    case cista::hash("tram"): return transport_mode::tram;
    case cista::hash("coach"): return transport_mode::coach;
    case cista::hash("rail"): return transport_mode::rail;
    case cista::hash("intercityRail"): return transport_mode::intercityRail;
    case cista::hash("urbanRail"): return transport_mode::urbanRail;
    case cista::hash("metro"): return transport_mode::metro;
    case cista::hash("air"): return transport_mode::air;
    case cista::hash("water"): return transport_mode::water;
    case cista::hash("cableway"): return transport_mode::cableway;
    case cista::hash("funicular"): return transport_mode::funicular;
    case cista::hash("snowAndIce"): return transport_mode::snowAndIce;
    case cista::hash("taxi"): return transport_mode::taxi;
    case cista::hash("selfDrive"): return transport_mode::selfDrive;
    default: return transport_mode::unknown;
  }
}

struct service_journey {
  std::uint32_t trip_nr_;
};

// Stop Point in Journey Pattern
using stop_point_in_journey_pattern_idx_t =
    cista::strong<std::uint32_t, struct _stop_point_in_journey_pattern_idx>;
using scheduled_stop_point_idx_t =
    cista::strong<std::uint32_t, struct _scheduled_stop_point_idx>;
using stop_place_idx_t = cista::strong<std::uint32_t, struct _stop_place_idx>;

struct stop_place {
  str_idx_t global_id_;
  str_idx_t name_;
  geo::latlng coord_;
};

struct stop_point_in_journey_pattern {
  ref_t<scheduled_stop_point_idx_t> scheduled_stop_point_;
};

template <typename Idx, typename T>
struct lookup {
  vector_map<Idx, T> storage_;
  string_store<Idx> lookup_;
};

// Mapping stops from ServiceJourney to stop_idx_t:
// TimetabledPassingTime.StopPointInJourneyPatternRef
// -> StopPointInJourneyPattern
// -> ScheduledStopPointRef
// -> PassengerStopAssignment [between ScheduledStopPointRef vs StopPlaceRef]
// -> StopPlaceRef
// -> stop_idx_t
struct intermediate {
  lookup<stop_point_in_journey_pattern_idx_t, stop_point_in_journey_pattern>
      stop_point_in_journey_pattern_;
  // lookup<scheduled_stop_point_idx_t, > x_;
  lookup<stop_place_idx_t, stop_place> stop_places_;
  string_store<str_idx_t> str_;
};

}  // namespace nigiri::loader::netex