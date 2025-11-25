#pragma once

namespace nigiri::loader::netex {

template <typename T>
using ref_t = std::variant<T, std::string>;

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

struct intermediate {

  std::vector <
};

}  // namespace nigiri::loader::netex