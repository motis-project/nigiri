#include "nigiri/clasz.h"

#include "cista/hash.h"

namespace nigiri {

clasz get_clasz(std::string_view s) {
  using cista::hash;
  switch (hash(s)) {
    case hash("Flug"): return clasz::kAir;
    case hash("Air"): return clasz::kAir;
    case hash("International Air"): return clasz::kAir;
    case hash("Domestic Air"): return clasz::kAir;
    case hash("Intercontinental Air"): return clasz::kAir;
    case hash("Domestic Scheduled Air"): return clasz::kAir;
    case hash("Shuttle Air"): return clasz::kAir;
    case hash("Intercontinental Charter Air"): return clasz::kAir;
    case hash("International Charter Air"): return clasz::kAir;
    case hash("Round-Trip Charter Air"): return clasz::kAir;
    case hash("Sightseeing Air"): return clasz::kAir;
    case hash("Helicopter Air"): return clasz::kAir;
    case hash("Domestic Charter Air"): return clasz::kAir;
    case hash("Schengen-Area Air"): return clasz::kAir;
    case hash("Airship"): return clasz::kAir;
    case hash("All Airs"): return clasz::kAir;

    // high speed
    case hash("High Speed Rail"): return clasz::kHighSpeed;
    case hash("ICE"): return clasz::kHighSpeed;
    case hash("THA"): return clasz::kHighSpeed;
    case hash("TGV"): return clasz::kHighSpeed;
    case hash("RJ"): return clasz::kHighSpeed;
    case hash("RJX"): return clasz::kHighSpeed;

    // range rail
    case hash("Long Distance Trains"): return clasz::kLongDistance;
    case hash("Inter Regional Rail"): return clasz::kLongDistance;
    case hash("Eurocity"): return clasz::kLongDistance;
    case hash("EC"): return clasz::kLongDistance;
    case hash("IC"): return clasz::kLongDistance;
    case hash("EX"): return clasz::kLongDistance;
    case hash("EXT"): return clasz::kLongDistance;
    case hash("D"): return clasz::kLongDistance;
    case hash("InterRegio"): return clasz::kLongDistance;
    case hash("Intercity"): return clasz::kLongDistance;

    // long range bus
    case hash("Coach"): return clasz::kCoach;
    case hash("International Coach"): return clasz::kCoach;
    case hash("National Coach"): return clasz::kCoach;
    case hash("Shuttle Coach"): return clasz::kCoach;
    case hash("Regional Coach"): return clasz::kCoach;
    case hash("Special Coach"): return clasz::kCoach;
    case hash("Sightseeing Coach"): return clasz::kCoach;
    case hash("Tourist Coach"): return clasz::kCoach;
    case hash("Commuter Coach"): return clasz::kCoach;
    case hash("All Coachs"): return clasz::kCoach;
    case hash("EXB"):
      return clasz::kCoach;  // long-distance bus

    // night trains
    case hash("Sleeper Rail"): return clasz::kNight;
    case hash("CNL"): return clasz::kNight;
    case hash("EN"): return clasz::kNight;
    case hash("Car Transport Rail"): return clasz::kNight;
    case hash("Lorry Transport Rail"): return clasz::kNight;
    case hash("Vehicle Transport Rail"): return clasz::kNight;
    case hash("AZ"): return clasz::kNight;
    case hash("NJ"): return clasz::kNight;

    // fast local trains
    case hash("RE"): return clasz::kRegionalFast;
    case hash("REX"): return clasz::kRegionalFast;
    case hash("IR"): return clasz::kRegionalFast;
    case hash("IRE"): return clasz::kRegionalFast;
    case hash("X"): return clasz::kRegionalFast;
    case hash("DPX"): return clasz::kRegionalFast;
    case hash("E"): return clasz::kRegionalFast;
    case hash("Sp"): return clasz::kRegionalFast;
    case hash("RegioExpress"): return clasz::kRegionalFast;
    case hash("TER"):
      return clasz::kRegionalFast;  // Transport express regional
    case hash("TE2"):
      return clasz::kRegionalFast;  // Transport express regional
    case hash("Cross-Country Rail"): return clasz::kRegionalFast;

    // local trains
    case hash("Railway Service"): return clasz::kRegional;
    case hash("Regional Rail"): return clasz::kRegional;
    case hash("Tourist Railway"): return clasz::kRegional;
    case hash("Rail Shuttle (Within Complex)"): return clasz::kRegional;
    case hash("Replacement Rail"): return clasz::kRegional;
    case hash("Special Rail"): return clasz::kRegional;
    case hash("Rack and Pinion Railway"): return clasz::kRegional;
    case hash("Additional Rail"): return clasz::kRegional;
    case hash("All Rails"): return clasz::kRegional;
    case hash("DPN"): return clasz::kRegional;
    case hash("R"): return clasz::kRegional;
    case hash("DPF"): return clasz::kRegional;
    case hash("RB"): return clasz::kRegional;
    case hash("Os"): return clasz::kRegional;
    case hash("Regionalzug"): return clasz::kRegional;
    case hash("RZ"): return clasz::kRegional;
    case hash("CC"): return clasz::kRegional;  // narrow-gauge mountain train
    case hash("PE"):
      return clasz::kRegional;  // Panorama Express

    // metro
    case hash("S"): return clasz::kMetro;
    case hash("S-Bahn"): return clasz::kMetro;
    case hash("SB"): return clasz::kMetro;
    case hash("Metro"): return clasz::kMetro;
    case hash("Schnelles Nachtnetz"): return clasz::kMetro;
    case hash("SN"):
      return clasz::kMetro;  // S-Bahn Nachtlinie

    // subway
    case hash("U"): return clasz::kSubway;
    case hash("STB"): return clasz::kSubway;
    case hash("M"): return clasz::kSubway;

    // street - car
    case hash("Tram"): return clasz::kTram;
    case hash("STR"): return clasz::kTram;
    case hash("Str"): return clasz::kTram;
    case hash("T"): return clasz::kTram;

    // bus
    case hash("Bus"): return clasz::kBus;
    case hash("B"): return clasz::kBus;
    case hash("BN"): return clasz::kBus;
    case hash("BP"): return clasz::kBus;
    case hash("CAR"): return clasz::kBus;
    case hash("KB"): return clasz::kBus;

    // ship
    case hash("Schiff"): return clasz::kShip;
    case hash("Fähre"): return clasz::kShip;
    case hash("BAT"): return clasz::kShip;  // "bateau"
    case hash("KAT"): return clasz::kShip;
    case hash("Ferry"): return clasz::kShip;
    case hash("Water Transport"): return clasz::kShip;
    case hash("International Car Ferry"): return clasz::kShip;
    case hash("National Car Ferry"): return clasz::kShip;
    case hash("Regional Car Ferry"): return clasz::kShip;
    case hash("Local Car Ferry"): return clasz::kShip;
    case hash("International Passenger Ferry"): return clasz::kShip;
    case hash("National Passenger Ferry"): return clasz::kShip;
    case hash("Regional Passenger Ferry"): return clasz::kShip;
    case hash("Local Passenger Ferry"): return clasz::kShip;
    case hash("Post Boat"): return clasz::kShip;
    case hash("Train Ferry"): return clasz::kShip;
    case hash("Road-Link Ferry"): return clasz::kShip;
    case hash("Airport-Link Ferry"): return clasz::kShip;
    case hash("Car High-Speed Ferry"): return clasz::kShip;
    case hash("Passenger High-Speed Ferry"): return clasz::kShip;
    case hash("Sightseeing Boat"): return clasz::kShip;
    case hash("School Boat"): return clasz::kShip;
    case hash("Cable-Drawn Boat"): return clasz::kShip;
    case hash("River Bus"): return clasz::kShip;
    case hash("Scheduled Ferry"): return clasz::kShip;
    case hash("Shuttle Ferry"): return clasz::kShip;
    case hash("All Water Transports"): return clasz::kShip;

    // other
    case hash("ZahnR"): return clasz::kOther;
    case hash("Schw-B"): return clasz::kOther;
    case hash("EZ"): return clasz::kOther;
    case hash("Taxi"): return clasz::kOther;
    case hash("ALT"): return clasz::kOther;  // "Anruflinientaxi"
    case hash("AST"): return clasz::kOther;  // "Anrufsammeltaxi"
    case hash("RFB"): return clasz::kOther;
    case hash("RT"): return clasz::kOther;
    case hash("Communal Taxi"): return clasz::kOther;
    case hash("Water Taxi"): return clasz::kOther;
    case hash("Rail Taxi"): return clasz::kOther;
    case hash("Bike Taxi"): return clasz::kOther;
    case hash("Licensed Taxi"): return clasz::kOther;
    case hash("Private Hire Vehicle"): return clasz::kOther;
    case hash("All Taxis"): return clasz::kOther;
    case hash("Self Drive"): return clasz::kOther;
    case hash("Hire Car"): return clasz::kOther;
    case hash("Hire Van"): return clasz::kOther;
    case hash("Hire Motorbike"): return clasz::kOther;
    case hash("Hire Cycle"): return clasz::kOther;
    case hash("All Self-Drive Vehicles"): return clasz::kOther;
    case hash("Car train"): return clasz::kOther;
    case hash("GB"): return clasz::kOther;  // ski lift / "funicular"?
    case hash("PB"): return clasz::kOther;  // also a ski lift(?)
    case hash("FUN"): return clasz::kOther;  // "funicular"
    case hash("Funicular"): return clasz::kOther;
    case hash("Telecabin"): return clasz::kOther;
    case hash("Cable Car"): return clasz::kOther;
    case hash("Chair Lift"): return clasz::kOther;
    case hash("Drag Lift"): return clasz::kOther;
    case hash("Small Telecabin"): return clasz::kOther;
    case hash("All Telecabins"): return clasz::kOther;
    case hash("All Funicular"): return clasz::kOther;
    case hash("Drahtseilbahn"): return clasz::kOther;
    case hash("Standseilbahn"): return clasz::kOther;
    case hash("Sesselbahn"): return clasz::kOther;
    case hash("Gondola): return claszle car"): return clasz::kOther;
    case hash("Aufzug"): return clasz::kOther;
    case hash("Elevator"): return clasz::kOther;
    case hash("ASC"): return clasz::kOther;
  }
  return clasz::kOther;
}

}  // namespace nigiri