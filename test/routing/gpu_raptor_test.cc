#pragma once
#include <random>
#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/lookup/get_transport.h"
#include "nigiri/routing/gpu_raptor_translator.h"
#include "../loader/hrd/hrd_timetable.h"
#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::routing;
using namespace nigiri::test_data::hrd_timetable;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;


std::filesystem::path project_root = std::filesystem::current_path().parent_path();
std::filesystem::path test_path_germany_zip(project_root / "test/routing/20240916_fahrplaene_gesamtdeutschland_gtfs.zip");
std::filesystem::path test_path_germany(project_root / "test/routing/20240916_fahrplaene_gesamtdeutschland_gtfs");
auto const german_dir_zip = zip_dir{test_path_germany_zip};
auto const german_dir = fs_dir{test_path_germany};

std::vector<std::basic_string_view<char>> get_locations(const timetable& tt) {
  std::vector<std::basic_string_view<char>> locations;
  locations.reserve(tt.n_locations()-9); //Die ersten neun Werte in tt.locations sind keine Locations.
  for (int i = 9; i < tt.n_locations(); ++i) {
    auto location_id = tt.locations_.get(location_idx_t{i}).id_;
    locations.push_back(location_id);
  }
  return locations;
}

double calculate_average(const std::vector<long long>& times) {
  if (times.empty()){
    return 0.0;
  }
  long long total = std::accumulate(times.begin(), times.end(), 0LL);
  return static_cast<double>(total) / times.size();
}
long long calculate_percentile(std::vector<long long>& times,double number) {
  std::sort(times.begin(), times.end());
  size_t idx = static_cast<size_t>(number * times.size());
  return times[idx];
}

std::pair<std::basic_string_view<char>, std::basic_string_view<char>> get_random_location_pair(const std::vector<std::basic_string_view<char>>& locations, std::mt19937& gen) {
  std::uniform_int_distribution<> dis(0, locations.size() - 1);

  std::basic_string_view<char> start_station = locations[dis(gen)];
  std::basic_string_view<char> end_station = locations[dis(gen)];

  return {start_station, end_station};
}

TEST(routing, gpu_benchmark) {
  timetable tt;
  std::cout << "Lade Fahrplan..." << std::endl;

  tt.date_range_ = {date::sys_days{2024_y / September / 25},
                    date::sys_days{2024_y / September / 26}}; //test_files_germany only available until December 14
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0}, german_dir_zip, tt);
  std::cout << "Fahrplan geladen." << std::endl;

  std::cout << "Finalisiere Fahrplan..." << std::endl;
  loader::finalize(tt);
  std::cout << "Fahrplan finalisiert." << std::endl;
  auto gtt = translate_tt_in_gtt(tt);
  constexpr int num_queries = 10000;
  std::vector<long long> cpu_times;
  std::vector<long long> gpu_times;
  int matched_queries = 0;

  std::random_device rd;
  unsigned int seed = rd(); //hier kann man anstatt sich random mit rd() einen Seed generien lassen auch direkt einen Seed angeben
  std::cout << "Verwendeter Seed: " << seed << std::endl;

  std::mt19937 gen(seed);
  auto locations = get_locations(tt);

  for (int i = 0; i < num_queries; ++i) {
    auto [start, end] = get_random_location_pair(locations,gen);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto const results_cpu = raptor_search(tt, nullptr,
                                           start, end,
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();
    std::stringstream ss_cpu, ss_gpu;
    ss_cpu << "\n";
    cpu_times.push_back(cpu_duration);
    for (auto const& x : results_cpu) {
      x.print(ss_cpu, tt);
      ss_cpu << "\n\n";
    }
    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto const results_gpu = raptor_search(tt, nullptr ,gtt,
                                           start, end,
                                           interval{unixtime_t{sys_days{2024_y / September / 25}} + 11_hours,
                                                    unixtime_t{sys_days{2024_y / September / 25}} + 13_hours},
                                           nigiri::direction::kBackward);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();
    gpu_times.push_back(gpu_duration);

    ss_gpu << "\n";
    for (auto const& x : results_gpu) {
      x.print(ss_gpu, tt);
      ss_gpu << "\n\n";
    }

    EXPECT_EQ(ss_cpu.str(), ss_gpu.str())
        << "Results differ for query " << i + 1 << ": " << start << " -> " << end;

    if (ss_cpu.str() == ss_gpu.str()) {
      matched_queries++;
    }

    if ((i + 1) % 10 == 0) {
      std::cout << "Bearbeitet: " << (i + 1) << " von " << num_queries << " Querys " << std::endl;
    }

  }
  double avg_cpu_time = calculate_average(cpu_times);
  double avg_gpu_time = calculate_average(gpu_times);
  long long cpu_99th = calculate_percentile(cpu_times,0.99);
  long long gpu_99th = calculate_percentile(gpu_times,0.99);
  long long cpu_90th = calculate_percentile(cpu_times,0.90);
  long long gpu_90th = calculate_percentile(gpu_times,0.90);
  long long cpu_50th = calculate_percentile(cpu_times,0.50);
  long long gpu_50th = calculate_percentile(gpu_times,0.50);

  std::cout << "Average CPU Time: " << avg_cpu_time << " microseconds\n";
  std::cout << "Average GPU Time: " << avg_gpu_time << " microseconds\n";
  std::cout << "99th Percentile CPU Time: " << cpu_99th << " microseconds\n";
  std::cout << "99th Percentile GPU Time: " << gpu_99th << " microseconds\n";
  std::cout << "90th Percentile CPU Time: " << cpu_90th << " microseconds\n";
  std::cout << "90th Percentile GPU Time: " << gpu_90th << " microseconds\n";
  std::cout << "50th Percentile CPU Time: " << cpu_50th << " microseconds\n";
  std::cout << "50th Percentile GPU Time: " << gpu_50th << " microseconds\n";
  std::cout << "Matched Queries: " << matched_queries << "/" << num_queries << "\n";
}