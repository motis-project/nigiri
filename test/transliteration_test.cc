#include "gtest/gtest.h"

#include <string>

#include "nigiri/loader/load.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace date;

namespace {

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
1,Укрзалізниця,https://www.uz.gov.ua/,Europe/Kyiv

# routes.txt
route_long_name,route_short_name,agency_id,route_type,route_id

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type,shape_dist_traveled

# trips.txt
route_id,service_id,shape_id,trip_id,wheelchair_accessible,bikes_allowed

# calendar_dates.txt
service_id,exception_type,date

# stops.txt
stop_name,stop_id,stop_lat,stop_lon
Київ-Пасажирський,2200005,50.440014,30.488482
Chełm,5100303,51.141315,23.4942
Львів-Головний,2200025,49.83993,23.993702

# feed_info.txt
feed_lang,feed_start_date,feed_end_date
uk,20190501,20190502
)";

constexpr auto const kGTFSNoLang = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
1,Укрзалізниця,https://www.uz.gov.ua/,Europe/Kyiv

# routes.txt
route_long_name,route_short_name,agency_id,route_type,route_id

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type,shape_dist_traveled

# trips.txt
route_id,service_id,shape_id,trip_id,wheelchair_accessible,bikes_allowed

# calendar_dates.txt
service_id,exception_type,date

# stops.txt
stop_name,stop_id,stop_lat,stop_lon
Київ-Пасажирський,2200005,50.440014,30.488482
Chełm,5100303,51.141315,23.4942
Львів-Головний,2200025,49.83993,23.993702
)";

constexpr auto const kGTFSmul = R"(
# feed_info.txt
feed_lang
mul

# agency.txt
agency_id,agency_name,agency_url,agency_timezone
1,Укрзалізниця,https://www.uz.gov.ua/,Europe/Kyiv

# routes.txt
route_long_name,route_short_name,agency_id,route_type,route_id

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type,shape_dist_traveled

# trips.txt
route_id,service_id,shape_id,trip_id,wheelchair_accessible,bikes_allowed

# calendar_dates.txt
service_id,exception_type,date

# stops.txt
stop_name,stop_id,stop_lat,stop_lon
Київ-Пасажирський,2200005,50.440014,30.488482
Chełm,5100303,51.141315,23.4942
Львів-Головний,2200025,49.83993,23.993702
)";

constexpr auto const kGTFSzh = R"(
# feed_info.txt
feed_lang
zh

# stops.txt
stop_name,stop_id,stop_lat,stop_lon
一面坡,YPB,45.059406,128.0753
一面坡北,YXB,45.083714,128.08044

# routes.txt
route_long_name,route_short_name,agency_id,route_type,route_id

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type,shape_dist_traveled

# trips.txt
route_id,service_id,shape_id,trip_id,wheelchair_accessible,bikes_allowed

# calendar_dates.txt
service_id,exception_type,date

# agency.txt
agency_id,agency_name,agency_url,agency_timezone
CR,中国国家铁路集团有限公司,https://www.12306.cn,Asia/Shanghai
)";
}  // namespace

std::string_view get_stop_translation(timetable const& tt,
                                      std::string_view id,
                                      std::string lang) {
  auto const loc_idx = tt.find(location_id{id, static_cast<source_idx_t>(0)});
  auto const translation_idx = tt.locations_.names_[*loc_idx];
  auto const s = tt.translate({{lang}}, translation_idx);
  return s;
}

TEST(transliteration, cyrillic) {
  auto const tt = loader::load(loader::mem_dir::read(kGTFS),
                               {2019_y / May / 1, 2019_y / May / 2});

  ASSERT_EQ(get_stop_translation(tt, "2200005", "uk-Latn"),
            "Kyyiv-Pasazhyrsʹkyy");
  ASSERT_EQ(get_stop_translation(tt, "5100303", "uk-Latn"),
            "Chełm");  // Make sure not to touch existing latin names
  ASSERT_EQ(get_stop_translation(tt, "2200025", "uk-Latn"), "Lʹviv-Holovnyy");
}

TEST(transliteration, zh) {
  auto const tt = loader::load(loader::mem_dir::read(kGTFSzh),
                               {2019_y / May / 1, 2019_y / May / 2});

  ASSERT_EQ(get_stop_translation(tt, "YPB", "zh-Latn"), "yī miàn pō");
}

TEST(transliteration, fallback) {
  auto const tt = loader::load(loader::mem_dir::read(kGTFS),
                               {2019_y / May / 1, 2019_y / May / 2});

  ASSERT_EQ(get_stop_translation(tt, "2200005", "en"), "Kyyiv-Pasazhyrsʹkyy");
  ASSERT_EQ(get_stop_translation(tt, "2200025", "sr"), "Львів-Головний");
}

TEST(transliteration, no_lang) {
  auto const tt = loader::load(loader::mem_dir::read(kGTFSNoLang),
                               {2019_y / May / 1, 2019_y / May / 2});

  ASSERT_EQ(get_stop_translation(tt, "2200005", "uk-Latn"),
            "Київ-Пасажирський");
  ASSERT_EQ(get_stop_translation(tt, "5100303", "uk-Latn"), "Chełm");
  ASSERT_EQ(get_stop_translation(tt, "2200025", "uk-Latn"), "Львів-Головний");
}

TEST(transliteration, mul) {
  auto const tt = loader::load(loader::mem_dir::read(kGTFSmul),
                               {2019_y / May / 1, 2019_y / May / 2});

  ASSERT_EQ(get_stop_translation(tt, "2200005", "uk-Latn"),
            "Київ-Пасажирський");
  ASSERT_EQ(get_stop_translation(tt, "5100303", "uk-Latn"), "Chełm");
  ASSERT_EQ(get_stop_translation(tt, "2200025", "uk-Latn"), "Львів-Головний");
}