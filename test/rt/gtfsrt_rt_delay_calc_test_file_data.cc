#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/logging.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/util.h"
#include "nigiri/timetable.h"

#include <cstdlib>

#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <regex>

#include "./util.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::rt;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_literals;
using namespace std::string_view_literals;

namespace {

date::sys_days extract_date(std::string const& s) {
  std::smatch match;
  if (std::regex_search(s, match, std::regex(R"((\d{4})-?(\d{2})-?(\d{2}))"))) {
    int y = std::stoi(match[1].str());
    int m = std::stoi(match[2].str());
    int d = std::stoi(match[3].str());
    return date::sys_days{date::year{y} / m / d};
  }
  return date::sys_days{date::year{1970} / 1 / 1};
}

}  // namespace


TEST(rt, gtfsrt_rt_delay_calc_file_data) {
  nigiri::s_verbosity = nigiri::log_lvl::error;

  auto const base_dir = std::filesystem::path{"test/test_data/gtfs_rt"};

  std::vector<std::filesystem::path> tu_pb_dirs;
  std::vector<std::filesystem::path> pb_dirs;
  std::vector<std::filesystem::path> zip_files;

  for (auto const& entry :
       std::filesystem::directory_iterator{"test/test_data/tripUpdates"}) {
    if (entry.is_directory()) {
      tu_pb_dirs.push_back(entry.path());
    }
  }

  for (auto const& entry : std::filesystem::directory_iterator{base_dir}) {
    if (entry.is_directory()) {
      pb_dirs.push_back(entry.path());
    } else if (entry.is_regular_file() && entry.path().extension() == ".zip") {
      zip_files.push_back(entry.path());
    }
  }

  std::ranges::sort(tu_pb_dirs);
  std::ranges::sort(pb_dirs);
  std::ranges::sort(zip_files);

  auto tts = hist_trip_times_storage{};
  auto dps = delay_prediction_storage{};
  auto simple_dps = delay_prediction_storage{};

  struct daily_stat {
    std::string day;
    std::chrono::nanoseconds intelligent_duration;
    std::chrono::nanoseconds simple_duration;
    uint64_t msgs;
    uint64_t simple_msgs;
  };
  std::vector<daily_stat> daily_stats;

  if (zip_files.empty()) {
    std::cerr << "No zip files found!\n";
    return;
  }

  std::unordered_map<std::string, std::vector<unixtime_t>> actual_delays;

  struct error_stat {
    long long total_abs_error_min{0};
    long long count{0};
  };
  std::map<int, error_stat> overall_intelligent_stats;
  std::map<int, error_stat> overall_simple_stats;

  for (unsigned long i = 0; i < pb_dirs.size(); ++i) {
    auto const& pb_dir = pb_dirs[i];
    auto const dir_name = pb_dir.filename().string();
    int y, m, d;
    if (sscanf(dir_name.c_str(), "%d-%d-%d", &y, &m, &d) != 3) {
      continue;
    }
    auto const dir_date = date::sys_days{date::year{y} / m / d};

    std::filesystem::path zip_file = zip_files.front();
    date::sys_days best_zip_date = date::sys_days{date::year{1970} / 1 / 1};

    for (auto const& zf : zip_files) {
      auto zdate = extract_date(zf.filename().string());
      if (zdate != date::sys_days{date::year{1970} / 1 / 1} &&
          zdate < dir_date && zdate >= best_zip_date) {
        zip_file = zf;
        best_zip_date = zdate;
      }
    }

    std::cout << "Processing " << pb_dir << " with " << zip_file << std::endl;

    // Load static timetable.
    timetable tt;
    register_special_stations(tt);
    tt.date_range_ = {dir_date - date::days{1}, dir_date + date::days{2}};
    load_timetable({}, source_idx_t{0}, loader::zip_dir{zip_file.string()}, tt);
    finalize(tt);
    auto actual_rtt = rt::create_rt_timetable(tt, dir_date);
    auto simple_rtt = rt::create_rt_timetable(tt, dir_date);
    auto intelligent_rtt = rt::create_rt_timetable(tt, dir_date);

    //
    //  get actual delays
    //

    if (i >= pb_dirs.size() - tu_pb_dirs.size()) {
      std::cout << "Extracting actual delays from GTFS-RT Trip Updates..."
                << std::endl;
      auto const j = i - (pb_dirs.size() - tu_pb_dirs.size());

      std::cout << "Processing " << tu_pb_dirs[j] << std::endl;

      std::vector<std::filesystem::path> tu_pb_files;
      for (auto const& entry :
           std::filesystem::recursive_directory_iterator{tu_pb_dirs[j]}) {
        if (entry.is_regular_file() && entry.path().extension() == ".pb") {
          tu_pb_files.push_back(entry.path());
        }
      }
      std::ranges::sort(tu_pb_files);

      int tu_file_counter = 0;
      int number_of_files = static_cast<int>(tu_pb_files.size());
      for (auto const& tu_pb_file : tu_pb_files) {
        if (tu_file_counter % 10 == 0) {
          std::cout << "  Processing tu-file " << tu_file_counter << "/"
                    << number_of_files << std::endl;
        }
        tu_file_counter++;

        std::ifstream ifs{tu_pb_file, std::ios::binary};
        std::string const content{std::istreambuf_iterator<char>{ifs},
                                  std::istreambuf_iterator<char>{}};

        transit_realtime::FeedMessage msg;
        msg.ParseFromString(content);

        auto const file_name = tu_pb_file.filename().string();

        for (auto const& entity : msg.entity()) {
          if (entity.has_trip_update() && entity.trip_update().has_trip() &&
              entity.trip_update().trip().has_trip_id() &&
              entity.trip_update().trip().has_start_date()) {
            auto const& tu = entity.trip_update();

            auto const [r, trip_idx] = gtfsrt_resolve_run(
                dir_date, tt, &actual_rtt, source_idx_t{0}, tu.trip(), "");

            if (!r.valid()) {
              continue;
            }

            auto const trip_key = tu.trip().trip_id() + ":" + tu.trip().start_date();
            if (actual_delays[trip_key].size() < r.stop_range_.size() * 2U) {
              actual_delays[trip_key].resize(r.stop_range_.size() * 2U,
                                             unixtime_t{});
            }

            for (auto const& stu : tu.stop_time_update()) {
              std::optional<uint32_t> next_stop;
              if (stu.has_stop_sequence()) {

                auto const seq_bucket = tt.trip_stop_seq_numbers_[trip_idx];
                auto const seq_numbers = loader::gtfs::stop_seq_number_range{
                    std::span<stop_idx_t const>{seq_bucket.begin(),
                                                seq_bucket.end()},
                    static_cast<stop_idx_t>(r.stop_range_.size())};

                auto const seq_it = utl::find(seq_numbers, stu.stop_sequence());
                if (seq_it != end(seq_numbers)) {
                  next_stop = static_cast<uint32_t>(
                      std::distance(begin(seq_numbers), seq_it));
                }
              }

              if (next_stop.has_value()) {
                auto const stop_idx = next_stop.value();
                if (stu.has_arrival() && stu.arrival().has_time()) {
                  actual_delays[trip_key][stop_idx * 2] =
                      unixtime_t{std::chrono::duration_cast<i32_minutes>(
                          std::chrono::seconds{stu.arrival().time()})};
                }
                if (stu.has_departure() && stu.departure().has_time()) {
                  actual_delays[trip_key][stop_idx * 2 + 1] =
                      unixtime_t{std::chrono::duration_cast<i32_minutes>(
                          std::chrono::seconds{stu.departure().time()})};
                }
              }
            }
          }
        }
      }
    }

    auto vtm = vehicle_trip_matching{};

    auto dp = delay_prediction{algorithm::kIntelligent,
                               hist_trip_mode::kSameDay,
                               1,
                               5,
                               &dps,
                               &tts,
                               &vtm,
                               true};

    auto simple_dp = delay_prediction{algorithm::kSimple,
                                      hist_trip_mode::kSameDay,
                                      1,
                                      5,
                                      &simple_dps,
                                      &tts,
                                      &vtm,
                                      true};

    std::vector<std::filesystem::path> pb_files;
    for (auto const& entry :
         std::filesystem::recursive_directory_iterator{pb_dir}) {
      if (entry.is_regular_file() && entry.path().extension() == ".pb") {
        pb_files.push_back(entry.path());
      }
    }
    std::ranges::sort(pb_files);

    auto day_intelligent_duration = std::chrono::nanoseconds{0};
    auto day_simple_duration = std::chrono::nanoseconds{0};
    uint64_t day_msgs = 0;
    uint64_t day_simple_msgs = 0;

    int loop_file_counter = 0;
    for (auto const& pb_file : pb_files) {
      if (loop_file_counter % 10 == 0) {
        std::cout << "  Processing file " << loop_file_counter << "/"
                  << pb_files.size() << std::endl;
      }
      loop_file_counter++;

      std::ifstream ifs{pb_file, std::ios::binary};
      std::string const content{std::istreambuf_iterator<char>{ifs},
                                std::istreambuf_iterator<char>{}};

      auto const tp_start = std::chrono::high_resolution_clock::now();
      gtfsrt_update_buf(tt, intelligent_rtt, source_idx_t{0}, "", content, &dp);
      auto const tp_end = std::chrono::high_resolution_clock::now();

      if (i >= pb_dirs.size() - tu_pb_dirs.size()) {
        auto const simple_tp_start = std::chrono::high_resolution_clock::now();
        gtfsrt_update_buf(tt, simple_rtt, source_idx_t{0}, "", content,
                          &simple_dp);
        auto const simple_tp_end = std::chrono::high_resolution_clock::now();
        day_simple_duration +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                simple_tp_end - simple_tp_start);
        ++day_simple_msgs;
      }
      day_intelligent_duration +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(tp_end -
                                                               tp_start);
      ++day_msgs;
    }

    daily_stats.push_back({dir_name, day_intelligent_duration,
                           day_simple_duration, day_msgs, day_simple_msgs});
    std::cout << "\n";

    //
    // Error and Stats - Calculation
    //

    auto calc_stats = [&](auto const& preds,
                          std::map<int, error_stat>& out_stats) {
      for (auto const& [trip_id, snaps] : preds) {
        auto it = actual_delays.find(trip_id);
        if (it == actual_delays.end()) {
          continue;
        }
        auto const& act = it->second;
        for (auto const& snap : snaps) {
          for (size_t stop_idx = 0;
               stop_idx < snap.stop_predictions.size() && stop_idx < act.size();
               ++stop_idx) {
            auto const& opt_pred = snap.stop_predictions[stop_idx];
            if (!opt_pred.has_value()) {
              continue;
            }
            if (act[stop_idx] == unixtime_t{}) {
              continue;
            }

            auto const p_ts = opt_pred.value();
            auto const a_ts = act[stop_idx];
            auto const m_ts = snap.measurement_time;

            auto const horizon_min =
                std::chrono::duration_cast<std::chrono::minutes>(a_ts - m_ts)
                    .count();
            if (horizon_min < 0) {
              continue;
            }

            int const bin = static_cast<int>(horizon_min / 5);
            auto const err_min = std::abs(
                std::chrono::duration_cast<std::chrono::minutes>(p_ts - a_ts)
                    .count());

            out_stats[bin].total_abs_error_min += err_min;
            out_stats[bin].count++;
          }
        }
      }
    };

    std::string const intelligent_file_name =
        "predicted_delays_intelligent_" + dir_name + ".txt";
    std::ofstream predicted_out{intelligent_file_name};
    if (predicted_out) {
      for (auto const& [trip_id, snaps] : dps.trip_delays) {
        predicted_out << trip_id << "\n";
        for (auto const& snap : snaps) {
          auto const meas_str =
              date::format("%Y-%m-%d %H:%M:%S", snap.measurement_time);
          predicted_out << "   " << meas_str << ": ";

          bool first = true;
          for (size_t i = 0; i < snap.stop_predictions.size(); ++i) {
            if (snap.stop_predictions[i].has_value() &&
                snap.stop_delays[i].has_value()) {
              if (!first) {
                predicted_out << ", ";
              }
              first = false;

              auto const ev_type = (i % 2 == 0) ? "A" : "D";
              auto const time_str =
                  date::format("%H%M%S", snap.stop_predictions[i].value());
              auto const seq = i / 2;

              predicted_out << seq << ev_type << ":"
                            << snap.stop_delays[i].value().count();
            }
          }
          predicted_out << "\n";
        }
      }
    }

    calc_stats(dps.trip_delays, overall_intelligent_stats);
    calc_stats(simple_dps.trip_delays, overall_simple_stats);

    dps.trip_delays.clear();
    simple_dps.trip_delays.clear();
    actual_delays.clear();
  }

  std::ofstream stat_out{"delay_prediction_stats.txt"};

  auto print_both = [&](auto const& f) {
    f(std::cout);
    f(stat_out);
  };

  print_both([&](std::ostream& out) {
    out << "Number of used VehiclePositions simple: " << simple_dps.n_vp
        << "\n";
    out << "Number of used VehiclePositions intelligent: " << dps.n_vp << "\n";
    out << "Number of used VehiclePositions K1: " << dps.n_vp_k1 << "\n";
    for (uint16_t i = 0; i < dps.n_jumped_over_stps_sgmts.size(); ++i) {
      if (dps.n_jumped_over_stps_sgmts[i] > 0) {
        out << "Number of " << i << " hops: " << dps.n_jumped_over_stps_sgmts[i]
            << "\n";
      }
    }
    out << "----------------------------------------\n";
    out << "Performance Statistics per Day:\n";
    for (auto const& stat : daily_stats) {
      if (stat.msgs > 0) {
        auto const avg_intelligent_time_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                stat.intelligent_duration)
                .count() /
            static_cast<double>(stat.msgs);
        out << "Day: " << stat.day << " | Total updates: " << stat.msgs
            << " | Avg intelligent processing time per update: "
            << avg_intelligent_time_ms << " ms\n";
      }
      if (stat.simple_msgs > 0) {
        auto const avg_simple_time_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                stat.simple_duration)
                .count() /
            static_cast<double>(stat.simple_msgs);
        out << "Day: " << stat.day
            << " | Total simple updates: " << stat.simple_msgs
            << " | Avg simple processing time per update: "
            << avg_simple_time_ms << " ms\n";
      }
    }
    out << "----------------------------------------\n";
  });

  auto print_stats = [](std::string const& name,
                        std::map<int, error_stat> const& stats,
                        std::ostream& out) {
    if (stats.empty()) {
      out << name << " Predictions: No data available.\n";
      return;
    }
    out << name << " Prediction Accuracy:\n";
    for (auto const& [bin, st] : stats) {
      auto const avg = st.count == 0
                           ? 0.0
                           : static_cast<double>(st.total_abs_error_min) /
                                 static_cast<double>(st.count);
      out << "  Start in " << (bin * 5) << " to " << ((bin + 1) * 5)
          << " min: " << avg << " min avg deviation (" << st.count
          << " predictions)\n";
    }
  };

  print_both([&](std::ostream& out) {
    print_stats("Simple", overall_simple_stats, out);
    out << "\n";
    print_stats("Intelligent", overall_intelligent_stats, out);
    out << "----------------------------------------\n";
  });
}