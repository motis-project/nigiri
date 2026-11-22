#include "gtest/gtest.h"

#include "utl/to_vec.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/load.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/timetable.h"
#include "nigiri/translations_view.h"
#include "nigiri/wikidata.h"

#include "../rt/util.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace date;
using namespace std::chrono_literals;

namespace {

constexpr auto const kRealFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
bvg,BVG,,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
S1,Alpha,52.5000,13.4000,0,
S2,Beta,52.5050,13.4000,0,
S3,Gamma,52.5100,13.4000,0,
S4,Delta,52.5150,13.4000,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
R1,bvg,M4,,0

# trips.txt
route_id,service_id,trip_id
R1,WD,T1

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
WD,1,1,1,1,1,1,1,20240101,20241231

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,10:00:00,10:00:00,S1,1
T1,10:02:00,10:02:00,S2,2
T1,10:04:00,10:04:00,S3,3
T1,10:06:00,10:06:00,S4,4
)";

constexpr auto const kMetadataFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone,agency_lang
Q99633,Berliner Verkehrsbetriebe,https://www.bvg.de,Etc/UTC,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type,route_url,route_color,wikidata
osm:r1,Q99633,M 4,"Tram M4: Alpha <=> Delta",0,https://bvg.de/m4,BE1414,https://www.wikidata.org/wiki/Q62512299

# stops.txt
stop_id,stop_name,platform_code,stop_lat,stop_lon,location_type,parent_station,wikidata
osm:a,Alpha,,52.50005,13.40005,0,osm:pa,
osm:pa,Alpha,,52.5001,13.4001,1,,Q111
osm:b,Beta,,52.50505,13.40005,0,,
osm:x,Extra,,52.5075,13.4060,0,,
osm:c,Gamma,,52.51005,13.40005,0,,
osm:d,Delta,,52.51505,13.40005,0,,

# trips.txt
route_id,service_id,trip_id,trip_headsign,trip_short_name,cars_allowed
osm:r1,,osm:r10,,M4,

# stop_times.txt
trip_id,departure_time,arrival_time,stop_id,stop_sequence
osm:r10,,,osm:a,0
osm:r10,,,osm:b,1
osm:r10,,,osm:x,2
osm:r10,,,osm:c,3
osm:r10,,,osm:d,4
osm:r10,,,osm:unknown,5

# translations.txt
table_name,field_name,language,record_id,translation
stops,stop_name,ja,osm:pa,アルファ
)";

constexpr auto const kDuplicateFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone,wikidata
vbb,BVG,https://www.bvg.de,Europe/Berlin,Q99633

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,wikidata
X1,Alpha,52.5000,13.4000,0,,Q111
X2,Beta,52.5050,13.4000,0,,
X3,Gamma,52.5100,13.4000,0,,
X4,Delta,52.5150,13.4000,0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type,route_color,wikidata
V1,vbb,M4,,0,BE1414,Q62512299

# trips.txt
route_id,service_id,trip_id,bikes_allowed
V1,WD,VT1,1

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
WD,1,1,1,1,1,1,1,20240101,20241231

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
VT1,10:00:00,10:00:00,X1,1
VT1,10:02:00,10:02:00,X2,2
VT1,10:04:00,10:04:00,X3,3
VT1,10:06:00,10:06:00,X4,4
)";

constexpr auto const kDateRange = interval{date::sys_days{2024_y / March / 1},
                                           date::sys_days{2024_y / March / 8}};

route_id_idx_t route_id(timetable const& tt,
                        source_idx_t const src,
                        std::string_view const id) {
  auto const& ids = tt.route_ids_[src];
  for (auto r = route_id_idx_t{0U}; r != route_id_idx_t{ids.ids_.size()}; ++r) {
    if (ids.ids_.get(r) == id) {
      return r;
    }
  }
  return route_id_idx_t::invalid();
}

std::string translation_in(timetable const& tt,
                           translation_idx_t const t,
                           std::string_view const lang) {
  for (auto const [l, text] : get_translation_view(tt, t)) {
    if (tt.languages_.get(l) == lang) {
      return std::string{text};
    }
  }
  return {};
}

}  // namespace

TEST(wikidata, parse) {
  auto tt = timetable{};
  EXPECT_FALSE(parse_wikidata_id(tt, "").valid());
  EXPECT_FALSE(parse_wikidata_id(tt, "osm:r1").valid());
  EXPECT_FALSE(parse_wikidata_id(tt, "Q").valid());
  EXPECT_FALSE(parse_wikidata_id(tt, "foo Q1").valid());
  EXPECT_FALSE(parse_wikidata_id(tt, "https://example.org/Q42/").valid());

  auto const q42 = parse_wikidata_id(tt, "Q42");
  EXPECT_EQ(42U, q42.item_);
  EXPECT_EQ(wikibase_idx_t{0U}, q42.wikibase_);
  EXPECT_EQ("https://www.wikidata.org/wiki/Q42", wikidata_url(tt, q42));
  EXPECT_EQ(q42, parse_wikidata_id(tt, "https://www.wikidata.org/wiki/Q42"));
  EXPECT_TRUE(tt.wikibase_urls_.empty());

  auto const other =
      parse_wikidata_id(tt, "https://wikibase.example.org/wiki/Item:Q7");
  EXPECT_EQ(7U, other.item_);
  EXPECT_EQ(wikibase_idx_t{1U}, other.wikibase_);
  EXPECT_EQ("https://wikibase.example.org/wiki/Item:Q7",
            wikidata_url(tt, other));
  EXPECT_EQ(other,
            parse_wikidata_id(tt, "https://wikibase.example.org/wiki/Item:Q7"));
  EXPECT_EQ(2U, tt.wikibase_urls_.size());
}

TEST(loader, enrich_from_metadata_feed) {
  auto const real = mem_dir::read(kRealFeed);
  auto const meta = mem_dir::read(kMetadataFeed);

  auto const plain = load(std::vector<dir_source>{{real}}, kDateRange);
  auto const tt = load(std::vector<dir_source>{{real}, {meta}}, kDateRange);

  auto const n_active = [](timetable const& x) {
    auto n = 0U;
    for (auto t = transport_idx_t{0U}; t != x.next_transport_idx(); ++t) {
      n += x.bitfields_[x.transport_traffic_days_[t]].any() ? 1U : 0U;
    }
    return n;
  };
  EXPECT_EQ(1U, n_active(plain));
  EXPECT_EQ(1U, n_active(tt));
  EXPECT_EQ(1U, to_idx(tt.next_transport_idx()));
  EXPECT_EQ(1U, tt.n_routes());

  auto const src = source_idx_t{0U};
  auto const r = route_id(tt, src, "R1");
  ASSERT_NE(route_id_idx_t::invalid(), r);
  auto const& ids = tt.route_ids_[src];
  EXPECT_EQ("M4", tt.get_default_translation(ids.route_id_short_names_[r]));
  // Names are not enriched: it is unclear which one is wanted.
  EXPECT_EQ("", tt.get_default_translation(ids.route_id_long_names_[r]));
  EXPECT_EQ("https://bvg.de/m4",
            tt.get_default_translation(ids.route_id_url_[r]));
  EXPECT_EQ(to_color("BE1414"), ids.route_id_colors_[r].color_);
  EXPECT_EQ("https://www.wikidata.org/wiki/Q62512299",
            wikidata_url(tt, ids.route_id_wikidata_[r]));

  auto const& provider = tt.providers_[ids.route_id_provider_[r]];
  EXPECT_EQ("https://www.bvg.de", tt.get_default_translation(provider.url_));
  EXPECT_EQ("https://www.wikidata.org/wiki/Q99633",
            wikidata_url(tt, provider.wikidata_));

  auto const s1 = tt.find({"S1", src});
  ASSERT_TRUE(s1.has_value());
  EXPECT_EQ("https://www.wikidata.org/wiki/Q111",
            wikidata_url(tt, tt.locations_.wikidata_[*s1]));
  EXPECT_EQ("Alpha", tt.get_default_translation(tt.locations_.names_[*s1]));
  EXPECT_EQ("アルファ", translation_in(tt, tt.locations_.names_[*s1], "ja"));

  auto const s2 = tt.find({"S2", src});
  ASSERT_TRUE(s2.has_value());
  EXPECT_FALSE(tt.locations_.wikidata_[*s2].valid());
}

TEST(loader, metadata_feed_requires_line_name_match) {
  auto const real = mem_dir::read(kRealFeed);
  auto meta_str = std::string{kMetadataFeed};
  meta_str.replace(meta_str.find("M 4,\""), 3U, "M5 ");
  meta_str.replace(meta_str.find(",M4,"), 4U, ",M5,");
  auto const meta = mem_dir::read(meta_str);

  auto const tt = load(std::vector<dir_source>{{real}, {meta}}, kDateRange);
  auto const r = route_id(tt, source_idx_t{0U}, "R1");
  EXPECT_FALSE(tt.route_ids_[source_idx_t{0U}].route_id_wikidata_[r].valid());
  EXPECT_EQ(color_t{0U},
            tt.route_ids_[source_idx_t{0U}].route_id_colors_[r].color_);
}

TEST(loader, metadata_feed_lines_with_same_stops) {
  // M5 is listed first and has the same stop sequence as M4.
  auto meta_str = std::string{kMetadataFeed};
  meta_str.replace(meta_str.find("osm:r1,Q99633,M 4"), 0U,
                   "osm:r5,Q99633,M5,,0,,00FF00,Q5\n");
  meta_str.replace(meta_str.find("osm:r1,,osm:r10"), 0U,
                   "osm:r5,,osm:r50,,M5,\n");
  auto i = 0U;
  for (auto const s : {"a", "b", "x", "c", "d"}) {
    meta_str.replace(meta_str.find("\n# translations.txt"), 0U,
                     fmt::format("osm:r50,,,osm:{},{}\n", s, i++));
  }
  auto const real = mem_dir::read(kRealFeed);
  auto const meta = mem_dir::read(meta_str);

  auto const tt = load(std::vector<dir_source>{{real}, {meta}}, kDateRange);
  auto const src = source_idx_t{0U};
  auto const r = route_id(tt, src, "R1");
  EXPECT_EQ("https://www.wikidata.org/wiki/Q62512299",
            wikidata_url(tt, tt.route_ids_[src].route_id_wikidata_[r]));
  EXPECT_EQ(to_color("BE1414"), tt.route_ids_[src].route_id_colors_[r].color_);
}

TEST(loader, enrich_from_duplicates) {
  // Each feed has information the other lacks.
  auto real_str = std::string{kRealFeed};
  real_str.replace(real_str.find("route_type\nR1,bvg,M4,,0"),
                   std::string_view{"route_type\nR1,bvg,M4,,0"}.size(),
                   "route_type,route_url\nR1,bvg,M4,,0,https://bvg.de/m4");
  auto const real = mem_dir::read(real_str);
  auto const dup = mem_dir::read(kDuplicateFeed);

  auto const tt =
      load(std::vector<dir_source>{{real}, {dup}}, kDateRange,
           {.merge_dupes_intra_src_ = false, .merge_dupes_inter_src_ = true});

  auto active = transport_idx_t::invalid();
  auto n_active = 0U;
  for (auto t = transport_idx_t{0U}; t != tt.next_transport_idx(); ++t) {
    if (tt.bitfields_[tt.transport_traffic_days_[t]].any()) {
      active = t;
      ++n_active;
    }
  }
  ASSERT_EQ(1U, n_active);

  auto const trip =
      tt.merged_trips_[tt.transport_to_trip_section_[active][0]][0];
  auto const src = tt.trip_id_src_[trip];
  auto const r = tt.trip_route_id_[trip];
  auto const& ids = tt.route_ids_[src];
  EXPECT_EQ(to_color("BE1414"), ids.route_id_colors_[r].color_);
  EXPECT_EQ("https://bvg.de/m4",
            tt.get_default_translation(ids.route_id_url_[r]));
  EXPECT_EQ("https://www.wikidata.org/wiki/Q62512299",
            wikidata_url(tt, ids.route_id_wikidata_[r]));
  EXPECT_EQ(
      "https://www.wikidata.org/wiki/Q99633",
      wikidata_url(tt, tt.providers_[ids.route_id_provider_[r]].wikidata_));

  auto const first_stop =
      stop{tt.route_location_seq_[tt.transport_route_[active]][0]}
          .location_idx();
  EXPECT_EQ("https://www.wikidata.org/wiki/Q111",
            wikidata_url(tt, tt.locations_.wikidata_[first_stop]));
  EXPECT_TRUE(
      tt.route_flags_[kBikesAllowed][to_idx(tt.transport_route_[active]) * 2U]);
}

namespace {

std::string replaced(std::string s,
                     std::string_view const from,
                     std::string_view const to) {
  s.replace(s.find(from), from.size(), to);
  return s;
}

transport_idx_t active_transport(timetable const& tt) {
  for (auto t = transport_idx_t{0U}; t != tt.next_transport_idx(); ++t) {
    if (tt.bitfields_[tt.transport_traffic_days_[t]].any()) {
      return t;
    }
  }
  return transport_idx_t::invalid();
}

trip_idx_t first_trip(timetable const& tt, transport_idx_t const t) {
  return tt.merged_trips_[tt.transport_to_trip_section_[t].front()].front();
}

}  // namespace

TEST(loader, enrich_route_type_and_clasz) {
  // The first feed's transport is absorbed by the second feed's one.
  auto const real = mem_dir::read(
      replaced(std::string{kRealFeed}, "R1,bvg,M4,,0", "R1,bvg,M4,,102"));
  auto const dup = mem_dir::read(
      replaced(std::string{kDuplicateFeed}, "V1,vbb,M4,,0,", "V1,vbb,M4,,2,"));
  auto const tt =
      load(std::vector<dir_source>{{real}, {dup}}, kDateRange,
           {.merge_dupes_intra_src_ = false, .merge_dupes_inter_src_ = true});

  auto const t = active_transport(tt);
  ASSERT_NE(transport_idx_t::invalid(), t);
  auto const [src, r] = std::pair{tt.trip_id_src_[first_trip(tt, t)],
                                  tt.trip_route_id_[first_trip(tt, t)]};
  EXPECT_EQ(source_idx_t{1U}, src);
  EXPECT_EQ(route_type_t{102U}, tt.route_ids_[src].route_id_type_[r]);
  EXPECT_EQ(clasz::kLongDistance, tt.route_clasz_[tt.transport_route_[t]]);
}

TEST(loader, enrichment_splits_route) {
  // Only VT1 has a duplicate with bikes allowed: VT2 keeps its route.
  auto const real =
      mem_dir::read(replaced(std::string{kRealFeed}, "trip_id\nR1,WD,T1",
                             "trip_id,bikes_allowed\nR1,WD,T1,1"));
  auto const dup = mem_dir::read(
      replaced(replaced(std::string{kDuplicateFeed}, "V1,WD,VT1,1",
                        "V1,WD,VT1,\nV1,WD,VT2,"),
               "VT1,10:06:00,10:06:00,X4,4",
               "VT1,10:06:00,10:06:00,X4,4\nVT2,11:00:00,11:00:00,X1,1\n"
               "VT2,11:02:00,11:02:00,X2,2\nVT2,11:04:00,11:04:00,X3,3\n"
               "VT2,11:06:00,11:06:00,X4,4"));
  auto const tt =
      load(std::vector<dir_source>{{real}, {dup}}, kDateRange,
           {.merge_dupes_intra_src_ = false, .merge_dupes_inter_src_ = true});

  ASSERT_EQ(transport_idx_t{2U}, tt.next_transport_idx());
  ASSERT_EQ(2U, tt.n_routes());
  EXPECT_NE(tt.is_flag_set(kBikesAllowed, route_idx_t{0U}),
            tt.is_flag_set(kBikesAllowed, route_idx_t{1U}));
}

TEST(loader, reservation_from_pickup_type) {
  auto const on_request = mem_dir::read(
      replaced(std::string{kRealFeed},
               "stop_id,stop_sequence\nT1,10:00:00,10:00:00,S1,1\n"
               "T1,10:02:00,10:02:00,S2,2\nT1,10:04:00,10:04:00,S3,3\n"
               "T1,10:06:00,10:06:00,S4,4",
               "stop_id,stop_sequence,pickup_type\n"
               "T1,10:00:00,10:00:00,S1,1,2\nT1,10:02:00,10:02:00,S2,2,2\n"
               "T1,10:04:00,10:04:00,S3,3,2\nT1,10:06:00,10:06:00,S4,4,2"));
  auto const dup = mem_dir::read(kDuplicateFeed);

  auto config = loader_config{};
  config.reservation_not_required_default_.fill(true);
  auto const reservation_required = [&](std::vector<dir const*> const& dirs) {
    auto const sources = utl::to_vec(dirs, [&](dir const* d) {
      return dir_source{.dir_ = *d, .loader_config_ = config};
    });
    auto const tt =
        load(sources, kDateRange,
             {.merge_dupes_intra_src_ = false, .merge_dupes_inter_src_ = true});
    auto const t = active_transport(tt);
    return t != transport_idx_t::invalid() &&
           !tt.is_flag_set(kReservationNotRequired, tt.transport_route_[t]);
  };

  EXPECT_TRUE(reservation_required({&on_request}));
  EXPECT_FALSE(reservation_required({&dup}));
  EXPECT_TRUE(reservation_required({&on_request, &dup}));
}

// The absorbed duplicate's route is removed: GTFS-RT updates for its trip id
// must still reach the surviving transport.
TEST(loader, rt_update_for_fully_absorbed_trip) {
  auto const real = mem_dir::read(kRealFeed);
  auto const dup = mem_dir::read(kDuplicateFeed);
  auto const tt =
      load(std::vector<dir_source>{{real}, {dup}}, kDateRange,
           {.merge_dupes_intra_src_ = false, .merge_dupes_inter_src_ = true});
  ASSERT_EQ(1U, tt.n_routes());
  ASSERT_EQ(1U, to_idx(tt.next_transport_idx()));

  auto const day = date::sys_days{2024_y / March / 4};
  for (auto const& [src, trip_id] : {std::pair{source_idx_t{0U}, "T1"},
                                     std::pair{source_idx_t{1U}, "VT1"}}) {
    auto rtt = rt::create_rt_timetable(tt, day);
    auto const msg =
        test::to_feed_msg({test::trip{.trip_id_ = trip_id,
                                      .delays_ = {{.seq_ = 3,
                                                   .ev_type_ = event_type::kArr,
                                                   .delay_minutes_ = 5}}}},
                          day + 9h);
    auto const stats = rt::gtfsrt_update_msg(tt, rtt, src, "tag", msg);
    EXPECT_EQ(1, stats.total_entities_success_) << trip_id;

    auto const resolved = rt::gtfsrt_resolve_run(
        day, tt, &rtt, src, msg.entity(0).trip_update().trip());
    auto const& r = resolved.first;
    ASSERT_TRUE(r.valid()) << trip_id;
    EXPECT_EQ(transport_idx_t{0U}, r.t_.t_idx_) << trip_id;
    auto const fr = rt::frun{tt, &rtt, r};
    EXPECT_EQ(5min, fr[2].time(event_type::kArr) -
                        fr[2].scheduled_time(event_type::kArr))
        << trip_id;
  }
}
