#pragma once

#include "nigiri/types.h"

#include "timetable.h"

namespace nigiri::rt {

using coord_seq_idx_t = cista::strong<std::uint64_t, struct _coord_seq_idx>;
using trip_time_data_idx_t =
    cista::strong<std::uint64_t, struct _trip_time_data_idx>;
using segment_idx_t = cista::strong<std::uint32_t, struct _segment_idx>;

struct vp_candidate {
  explicit vp_candidate(rt::run const& r, std::uint32_t const total_length)
      : r_{r}, total_length_{total_length} {}

  friend bool operator<(vp_candidate const& a, vp_candidate const& b) {
    return a.score_ > b.score_ ||
           (a.score_ == b.score_ && a.total_length_ < b.total_length_);
  }

  friend bool operator==(vp_candidate const& a, vp_candidate const& b) {
    return a.score_ == b.score_ && a.total_length_ == b.total_length_;
  }

  void finish_stop() {
    score_ += local_best_;
    local_best_ = 0U;
  }

  run r_;
  std::uint32_t score_{0U};
  std::uint32_t local_best_{0U};
  std::uint32_t total_length_{};
};

struct vehicle_trip_matching {

  void clean_up();

  hash_map<vehicle_idx_t, run> vehicle_idx_run_;
  hash_map<vehicle_idx_t, vector<location_idx_t>> vehicle_idx_known_stop_locs_;
  hash_map<vehicle_idx_t, std::chrono::sys_seconds> vehicle_idx_last_access_;

  // Vehicle access: external vehicle id -> internal vehicle index
  vector<pair<vehicle_id_idx_t, vehicle_idx_t>> vehicle_id_to_idx_;

  // Storage for vehicle id strings + source
  vecvec<vehicle_id_idx_t, char> vehicle_id_strings_;
  vector_map<vehicle_id_idx_t, source_idx_t> vehicle_id_src_;

  std::chrono::sys_seconds last_cleanup{
      std::chrono::time_point_cast<std::chrono::seconds>(
          std::chrono::system_clock::now())};
};

struct key {
  transport_idx_t t_idx;
  source_idx_t source_idx;
};

struct trip_seg_data {
  segment_idx_t seg_idx;
  double progress;
  unixtime_t timestamp;
  geo::latlng position;

  bool operator==(trip_seg_data const& tsd) const {
    return seg_idx == tsd.seg_idx && progress == tsd.progress &&
           timestamp == tsd.timestamp && position == tsd.position;
  }
};

struct trip_time_data {
  trip_time_data(unixtime_t start_time, vector<trip_seg_data> seg_data, uint32_t n_stops) {
    start_timestamp = start_time;
    seg_data_ = std::move(seg_data);
    stop_durations_.resize(n_stops-1);
    segment_durations_.resize(n_stops-1);
  }
  unixtime_t start_timestamp;
  vector<trip_seg_data> seg_data_;
  vector<duration_t> stop_durations_;
  vector<duration_t> segment_durations_;
};

struct hist_trip_times_storage {
  explicit hist_trip_times_storage()
      : coord_seq_idx_ttd_{mm_paged_vecvec_helper<coord_seq_idx_t,
                                                  trip_time_data_idx_t>::data_t{
                               mm_vec<trip_time_data_idx_t>{
                                   cista::mmap{"hist_trip_time_data.bin"}}},
                           mm_vec<cista::page<std::uint64_t, std::uint32_t>>{
                               cista::mmap{"hist_trip_time_idx.bin"}}} {}

  hash_map<key, coord_seq_idx_t> cs_key_coord_seq_;
  paged_vecvec<coord_seq_idx_t, location_idx_t> coord_seq_idx_coord_seq_;

  mm_paged_vecvec<coord_seq_idx_t, trip_time_data_idx_t> coord_seq_idx_ttd_;
  vector_map<trip_time_data_idx_t, trip_time_data> ttd_idx_trip_time_data_;

  // check if key already exists
  // if not: check if similar enough coord_seq exists (find_duplicates())
  // if not: create new index and add entries data structures
  coord_seq_idx_t match_trip_to_coord_seq(timetable const&,
                                          key,
                                          vector<location_idx_t>);

  std::tuple<segment_idx_t, double, geo::latlng> get_segment_progress(
      timetable const&, geo::latlng, coord_seq_idx_t);

  static duration_t get_remaining_time_till_next_stop(trip_seg_data const*,
                                                      trip_time_data const*);

  void print(std::ostream& out) const;

  friend std::ostream& operator<<(std::ostream& out,
                                  hist_trip_times_storage const& tts);
};

struct trip_delay_pred {
  // filter gain
  double filter_gain;
  // gain loop
  double gain_loop;
  // Filter Error
  double error;
  // direct predecessors trip_time_data
  vector<trip_time_data_idx_t> predecessors_;
  // historic trips trip_time_data
  vector<trip_time_data_idx_t> hist_trips_;
  // average historic stop times
  vector<duration_t> hist_avg_stop_durations_;
  // average historic segment times
  vector<duration_t> hist_avg_segment_durations_;
};

struct delay_prediction_storage {
  hash_map<key, trip_delay_pred> key_trip_delay_;

  trip_delay_pred& get_or_create_kalman(
      key, unixtime_t, uint32_t, uint32_t, uint32_t, hist_trip_times_storage*);

  static pair<vector<duration_t>, vector<duration_t>>
  get_avg_stop_segment_durations(hist_trip_times_storage*,
                                 vector<trip_time_data_idx_t>);

  static duration_t get_avg_duration(vector<duration_t> const&);

  void print(std::ostream& out) const;

  friend std::ostream& operator<<(std::ostream& out,
                                  delay_prediction_storage const& tts);
};

}  // namespace nigiri::rt
