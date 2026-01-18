#include "nigiri/clasz.h"

#include "cista/hash.h"

#include "utl/verify.h"

#include "nigiri/logging.h"

namespace nigiri {

clasz get_clasz(std::string_view s) {
  using cista::hash;
  switch (hash(s)) {
    case hash("Flug"):
    case hash("Air"):
    case hash("International Air"):
    case hash("Domestic Air"):
    case hash("Intercontinental Air"):
    case hash("Domestic Scheduled Air"):
    case hash("Shuttle Air"):
    case hash("Intercontinental Charter Air"):
    case hash("International Charter Air"):
    case hash("Round-Trip Charter Air"):
    case hash("Sightseeing Air"):
    case hash("Helicopter Air"):
    case hash("Domestic Charter Air"):
    case hash("Schengen-Area Air"):
    case hash("Airship"): [[fallthrough]];
    case hash("All Airs"): return clasz::kAir;

    // high speed
    case hash("High Speed Rail"):
    case hash("ICE"):
    case hash("THA"):
    case hash("TGV"):
    case hash("RJ"): [[fallthrough]];
    case hash("RJX"): return clasz::kHighSpeed;

    // range rail
    case hash("Long Distance Trains"):
    case hash("Inter Regional Rail"):
    case hash("Eurocity"):
    case hash("EC"):
    case hash("IC"):
    case hash("EX"):
    case hash("EXT"):
    case hash("D"):
    case hash("InterRegio"): [[fallthrough]];
    case hash("Intercity"): return clasz::kLongDistance;

    // long range bus
    case hash("Coach"):
    case hash("International Coach"):
    case hash("National Coach"):
    case hash("Shuttle Coach"):
    case hash("Regional Coach"):
    case hash("Special Coach"):
    case hash("Sightseeing Coach"):
    case hash("Tourist Coach"):
    case hash("Commuter Coach"):
    case hash("All Coachs"): [[fallthrough]];
    case hash("EXB"):
      return clasz::kCoach;  // long-distance bus

    // night trains
    case hash("Sleeper Rail"):
    case hash("CNL"):
    case hash("EN"):
    case hash("Car Transport Rail"):
    case hash("Lorry Transport Rail"):
    case hash("Vehicle Transport Rail"):
    case hash("AZ"): [[fallthrough]];
    case hash("NJ"): return clasz::kNight;

    // fast local trains
    case hash("RE"):
    case hash("REX"):
    case hash("IR"):
    case hash("IRE"):
    case hash("X"):
    case hash("DPX"):
    case hash("E"):
    case hash("Sp"):
    case hash("RegioExpress"):
    case hash("TER"):  // Transport express regional
    case hash("TE2"): [[fallthrough]];  // Transport express regional
    case hash("Cross-Country Rail"): return clasz::kRegionalFast;

    // local trains
    case hash("Railway Service"):
    case hash("Regional Rail"):
    case hash("Tourist Railway"):
    case hash("Rail Shuttle (Within Complex)"):
    case hash("Replacement Rail"):
    case hash("Special Rail"):
    case hash("Rack and Pinion Railway"):
    case hash("Additional Rail"):
    case hash("All Rails"):
    case hash("DPN"):
    case hash("R"):
    case hash("DPF"):
    case hash("RB"):
    case hash("Os"):
    case hash("Regionalzug"):
    case hash("RZ"):
    case hash("CC"): [[fallthrough]];  // narrow-gauge mountain train
    case hash("PE"):
      return clasz::kRegional;  // Panorama Express

    // metro
    case hash("S"):
    case hash("S-Bahn"):
    case hash("SB"):
    case hash("Metro"):
    case hash("Schnelles Nachtnetz"): [[fallthrough]];
    case hash("SN"):
      return clasz::kSuburban;  // S-Bahn Nachtlinie

    // subway
    case hash("U"):
    case hash("STB"): [[fallthrough]];
    case hash("M"): return clasz::kSubway;

    // street - car
    case hash("Tram"):
    case hash("STR"):
    case hash("Str"): [[fallthrough]];
    case hash("T"): return clasz::kTram;

    // bus
    case hash("Bus"):
    case hash("B"):
    case hash("BN"):
    case hash("BP"):
    case hash("CAR"): [[fallthrough]];
    case hash("KB"): return clasz::kBus;

    // ship
    case hash("Schiff"):
    case hash("Fähre"):
    case hash("BAT"):  // "bateau"
    case hash("KAT"):
    case hash("Ferry"):
    case hash("Water Transport"):
    case hash("International Car Ferry"):
    case hash("National Car Ferry"):
    case hash("Regional Car Ferry"):
    case hash("Local Car Ferry"):
    case hash("International Passenger Ferry"):
    case hash("National Passenger Ferry"):
    case hash("Regional Passenger Ferry"):
    case hash("Local Passenger Ferry"):
    case hash("Post Boat"):
    case hash("Train Ferry"):
    case hash("Road-Link Ferry"):
    case hash("Airport-Link Ferry"):
    case hash("Car High-Speed Ferry"):
    case hash("Passenger High-Speed Ferry"):
    case hash("Sightseeing Boat"):
    case hash("School Boat"):
    case hash("Cable-Drawn Boat"):
    case hash("River Bus"):
    case hash("Scheduled Ferry"):
    case hash("Shuttle Ferry"): [[fallthrough]];
    case hash("All Water Transports"): return clasz::kShip;

    // other
    case hash("ZahnR"):
    case hash("Schw-B"):
    case hash("EZ"):
    case hash("Taxi"):
    case hash("ALT"):  // "Anruflinientaxi"
    case hash("AST"):  // "Anrufsammeltaxi"
    case hash("RFB"):
    case hash("RT"):
    case hash("Communal Taxi"):
    case hash("Water Taxi"):
    case hash("Rail Taxi"):
    case hash("Bike Taxi"):
    case hash("Licensed Taxi"):
    case hash("Private Hire Vehicle"):
    case hash("All Taxis"):
    case hash("Self Drive"):
    case hash("Hire Car"):
    case hash("Hire Van"):
    case hash("Hire Motorbike"):
    case hash("Hire Cycle"):
    case hash("All Self-Drive Vehicles"):
    case hash("Car train"):
    case hash("GB"):  // ski lift / "funicular"?
    case hash("PB"):  // also a ski lift(?)
    case hash("FUN"):  // "funicular"
    case hash("Funicular"):
    case hash("Telecabin"):
    case hash("Cable Car"):
    case hash("Chair Lift"):
    case hash("Drag Lift"):
    case hash("Small Telecabin"):
    case hash("All Telecabins"):
    case hash("All Funicular"):
    case hash("Drahtseilbahn"):
    case hash("Standseilbahn"):
    case hash("Sesselbahn"):
    case hash("Gondola"):
    case hash("Aufzug"):
    case hash("Elevator"): [[fallthrough]];
    case hash("ASC"): return clasz::kOther;
    default:
      log(log_lvl::error, "loader.hrd.clasz", "cannot assign {}", s);
      return clasz::kOther;
  }
}

clasz to_clasz(std::string_view s) {
  switch (cista::hash(s)) {
    case cista::hash("AIR"): return clasz::kAir;
    case cista::hash("HIGHSPEED"): return clasz::kHighSpeed;
    case cista::hash("LONGDISTANCE"): return clasz::kLongDistance;
    case cista::hash("COACH"): return clasz::kCoach;
    case cista::hash("NIGHT"): return clasz::kNight;
    case cista::hash("REGIONALFAST"): return clasz::kRegionalFast;
    case cista::hash("REGIONAL"): return clasz::kRegional;
    case cista::hash("METRO"): return clasz::kSuburban;
    case cista::hash("SUBURBAN"): return clasz::kSuburban;
    case cista::hash("SUBWAY"): return clasz::kSubway;
    case cista::hash("TRAM"): return clasz::kTram;
    case cista::hash("BUS"): return clasz::kBus;
    case cista::hash("SHIP"): return clasz::kShip;
    case cista::hash("OTHER"): return clasz::kOther;
  }
  throw utl::fail("{} is not a valid clasz", s);
}

}  // namespace nigiri
