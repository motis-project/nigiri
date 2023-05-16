#pragma once

#include "nigiri/loader/hrd/parser_config.h"

namespace nigiri::test_data::hrd_timetable {

//  0     1     2     3     4     5
//  A  -  B  -  C  -  D  -  E  -  F
//    100 - 110 - 111 - 011 - 001

// 28.03.  100 [0, 3] A - B - C - D
// 29.03.  010 [1, 4]     B - C - D - E
// 30.03.  001 [2, 5]         C - D - E - F

constexpr interval<std::chrono::sys_days> full_period() {
  using namespace date;
  constexpr auto const from = (2020_y / March / 28).operator sys_days();
  constexpr auto const to = (2020_y / March / 31).operator sys_days();
  return {from, to};
}

constexpr auto const basic_info_file_content = R"(26.03.2020
02.04.2020
Fahrplan 2020$29.03.2020 03:15:02$5.20.39$INFO+
)";

constexpr auto const stations_file_content = R"(
0000001     A
0000002     B
0000003     C
0000004     D
0000005     E
0000006     F
0000007     G
0000008     H
0000009     I
0000010     F_META
)";

constexpr auto const station_metabhf_content = R"(
0000010: 0000006
)";

constexpr auto const station_geo_file_content = R"(
0000001  32.034466  54.798343 A
0000002  34.317551  55.197393 B
0000003  36.579810  56.376671 C
0000004  38.579810  57.276672 D
0000005  40.579810  58.176673 E
0000006  41.579810  59.076673 F
0000007  42.559810  61.066673 G
0000008  43.469810  62.056673 H
0000009  44.379810  63.046673 I
0000007  41.579799  59.076849 F_META
)";

// 000001 = C86 = 11|00 1000 0|110 = 28.03
// 000002 = CC6 = 11|00 1100 0|110 = 28.03, 29.03
// 000003 = CE6 = 11|00 1110 0|110 = 28.03, 29.03, 30.03
// 000004 = C66 = 11|00 0110 0|110 =        29.03, 30.03
// 000005 = C26 = 11|00 0010 0|110 =               30.03
constexpr auto const bitfields_file_content = R"(
000001 C86
000002 CC6
000003 CE6
000004 C66
000005 C26
)";

constexpr auto const timezones_file_content = R"(
0000000 +0100 +0200 29032020 0200 25102020 0300 +0200 28032021 0200 31102021 0300
)";

constexpr auto const categories_file_content = R"(
ICE  0 A 0  ICE       2   Intercity-Express
IC   1 B 0  IC        2   Intercity
RE   3 C 0  RE        0 N Regional-Express
)";

constexpr auto const providers_file_content = R"(
00001 K '---' L 'DB AG' V 'Deutsche Bahn AG'
00001 : 80____
)";

constexpr auto const tracks_file_content = R"(
0000004 00815 80____ 1        2207 000001
0000004 00815 80____ 2        2207 000004
)";

inline loader::mem_dir base() {
  using namespace loader::hrd;
  auto const& b = hrd_5_20_26.core_data_;
  auto const& r = hrd_5_20_26.required_files_;
  return {{{(b / r[ATTRIBUTES][0]), ""},
           {(b / r[STATIONS][0]), stations_file_content},
           {(b / r[COORDINATES][0]), station_geo_file_content},
           {(b / r[BITFIELDS][0]), bitfields_file_content},
           {(b / r[TRACKS][0]), tracks_file_content},
           {(b / r[INFOTEXT][0]), ""},
           {(b / r[BASIC_DATA][0]), basic_info_file_content},
           {(b / r[CATEGORIES][0]), categories_file_content},
           {(b / r[DIRECTIONS][0]), ""},
           {(b / r[PROVIDERS][0]), providers_file_content},
           {(b / r[THROUGH_SERVICES][0]), ""},
           {(b / r[MERGE_SPLIT_SERVICES][0]), ""},
           {(b / r[TIMEZONES][0]), timezones_file_content},
           {(b / r[FOOTPATHS][0]), station_metabhf_content}}};
}

inline loader::mem_dir files() {
  constexpr auto const service_file_content = R"(
*Z 01337 80____       002 120                             %
*A VE 0000001 0000002 000001                              %
*A VE 0000002 0000003 000002                              %
*A VE 0000003 0000004 000003                              %
*A VE 0000004 0000005 000004                              %
*A VE 0000005 0000006 000005                              %
*G ICE 0000001 0000006                                    %
0000001 A                            02200                %
0000002 B                     02301  02302                %
0000003 C                     02704  02805                %
0000004 D                     04506  04607                %
0000005 E                     05008  05009                %
0000006 F                     05110                       %
*Z 00815 80____                                           %
*A VE 0000007 0000006 000003                              %
*G RE  0000007 0000006                                    %
0000007 G                            00230                %
0000003 C                     00320  00405                %
0000004 D                     02106  02207                %
0000005 E                     02608  02609                %
0000006 F                     02710                       %
*Z 03374 80____                                           %
*A VE 0000008 0000006 000003                              %
*G IC  0000008 0000006                                    %
0000008 H                            00030                %
0000007 G                     00043  00045                %
0000009 I                     02524  02525                %
0000005 E                     02555  02709                %
0000006 F                     02710                       %
)";
  auto const& f = loader::hrd::hrd_5_20_26.fplan_;
  return base().add({(f / "services.101"), service_file_content});
}

inline loader::mem_dir files_simple() {
  constexpr auto const services_simple = R"(
*Z 01337 80____       048 030                             %
*A VE 0000001 0000002 000005                              %
*G RE  0000001 0000002                                    %
0000001 A                            00230                %
0000002 B                     00330                       %
*Z 07331 80____       092 015                             %
*A VE 0000002 0000001 000005                              %
*G RE  0000002 0000001                                    %
0000002 B                            00230                %
0000001 A                     00330                       %
)";
  auto const& f = loader::hrd::hrd_5_20_26.fplan_;
  return base().add({(f / "services.101"), services_simple});
}

inline loader::mem_dir files_abc() {
  constexpr auto const services_abc = R"(
*Z 01337 80____       048 030                             %
*A VE 0000001 0000002 000005                              %
*G RE  0000001 0000002                                    %
0000001 A                            00230                %
0000002 B                     00330                       %
*Z 07331 80____       092 015                             %
*A VE 0000002 0000003 000005                              %
*G RE  0000002 0000003                                    %
0000002 B                            00230                %
0000003 C                     00330                       %
)";
  return test_data::hrd_timetable::base().add(
      {loader::hrd::hrd_5_20_26.fplan_ / "services.101", services_abc});
}

}  // namespace nigiri::test_data::hrd_timetable
