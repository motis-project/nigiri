#include "nigiri/delay_prediction.h"

#include <cfloat>

#include "nigiri/for_each_meta.h"
#include "nigiri/types.h"

namespace nigiri::rt {

void vehicle_trip_matching::clean_up() {
  auto const now = std::chrono::time_point_cast<std::chrono::seconds>(
      std::chrono::system_clock::now());

  std::erase_if(vehicle_idx_last_access_, [&](auto const& i) {
    auto const& [idx, la] = i;
    if (now - la > std::chrono::hours{48}) {
      vehicle_idx_run_.erase(idx);
      vehicle_idx_known_stop_locs_.erase(idx);
      return true;
    }
    return false;
  });
  last_cleanup = now;
}

// check if key already exists
// if not: check if similar enough coord_seq exists (find_duplicates())
// if not: create new index and add entries to data structures
coord_seq_idx_t hist_trip_times_storage::match_trip_to_coord_seq(
    timetable const& tt, key k, vector<location_idx_t> coord_seq) {

  if (cs_key_coord_seq_.contains(k)) {
    // key already exists
    return cs_key_coord_seq_[k];
  }

  for (coord_seq_idx_t idx{0}; idx < coord_seq_idx_coord_seq_.size(); ++idx) {
    auto const& bucket = coord_seq_idx_coord_seq_[idx];

    if (bucket.size() != coord_seq.size()) {
      continue;
    }

    for (ulong i = 0; i < bucket.size(); ++i) {
      if (!routing::matches(tt, routing::location_match_mode::kEquivalent,
                            bucket[i], coord_seq[i])) {
        break;
      }
      if (i == bucket.size() - 1) {
        // coord_seq already exists
        // create new [key, coord_seq_idx] entry
        cs_key_coord_seq_.emplace(k, idx);
        return idx;
      }
    }
  }
  // create new coord_seq_idx and new coord_seq and entries
  coord_seq_idx_coord_seq_.emplace_back(coord_seq);
  coord_seq_idx_t new_idx{coord_seq_idx_coord_seq_.size() - 1};
  cs_key_coord_seq_.emplace(k, new_idx);
  coord_seq_idx_ttd_.emplace_back(vector<trip_time_data_idx_t>{});
  return new_idx;
}

std::tuple<segment_idx_t, double, geo::latlng>
hist_trip_times_storage::get_segment_progress(timetable const& tt,
                                              geo::latlng vehicle_position,
                                              coord_seq_idx_t coord_seq_idx) {

  auto const app_dist_lng_deg_vp =
      geo::approx_distance_lng_degrees(vehicle_position);
  std::pair closest = {geo::latlng{0, 0}, DBL_MAX};

  auto segment_from = coord_seq_idx_coord_seq_[coord_seq_idx].begin();
  for (auto segment_to = coord_seq_idx_coord_seq_[coord_seq_idx].begin() + 1;
       segment_to != coord_seq_idx_coord_seq_[coord_seq_idx].end();
       ++segment_to) {

    auto const segment_to_test = geo::approx_closest_on_segment(
        vehicle_position, tt.locations_.coordinates_[*segment_from],
        tt.locations_.coordinates_[*segment_to], app_dist_lng_deg_vp);

    if (closest.second < segment_to_test.second) {
      break;
    }

    closest = segment_to_test;
    ++segment_from;
  }

  auto const adld = geo::approx_distance_lng_degrees(closest.first);

  auto const segment_from_pos = tt.locations_.coordinates_[*segment_from - 1];
  auto const segment_to_pos = tt.locations_.coordinates_[*segment_from];
  auto const progress =
      geo::approx_squared_distance(closest.first, segment_from_pos, adld) /
      geo::approx_squared_distance(segment_to_pos, segment_from_pos, adld);

  auto const nearest_stop = progress < 0.5 ? segment_from_pos : segment_to_pos;

  return {
      static_cast<segment_idx_t>(std::distance(
          begin(coord_seq_idx_coord_seq_[coord_seq_idx]), segment_from - 1)),
      progress, nearest_stop};
}

duration_t hist_trip_times_storage::get_remaining_time_till_next_stop(
    trip_seg_data const* tsd, trip_time_data const* ttd) {
  auto last_tsd_before_stop =
      std::find_if(ttd->seg_data_.rbegin(), ttd->seg_data_.rend(),
                   [tsd](auto const check_if_last_tsd) {
                     return tsd->seg_idx == check_if_last_tsd.seg_idx;
                   });

  return last_tsd_before_stop->timestamp - tsd->timestamp;
}

void hist_trip_times_storage::print(std::ostream& out) const {
  out << "\ncs_key_coord_seq_:\n";
  for (auto const& [key, coord_seq_idx] : cs_key_coord_seq_) {
    out << "Key: Source: " << key.source_idx << " Transport: " << key.t_idx
        << "\nCoord_seq_Idx: " << coord_seq_idx << "\n";
  }

  out << "\ncoord_seq_idx_coord_seq_:";
  for (coord_seq_idx_t idx{0}; idx < coord_seq_idx_coord_seq_.size(); ++idx) {
    out << "\nCoord_seq_Idx: " << idx << "\nLocation_Sequence: ";
    for (auto loc_idx : coord_seq_idx_coord_seq_[idx]) {
      out << loc_idx << ",";
    }
  }

  out << "\n\ncoord_seq_idx_ttd_:";
  for (coord_seq_idx_t idx{0}; idx < coord_seq_idx_ttd_.size(); ++idx) {
    out << "\nCoord_seq_Idx: " << idx << "\nTrip_Time_Data_Idxs: ";
    for (auto loc_idx : coord_seq_idx_ttd_[idx]) {
      out << loc_idx << ",";
    }
  }

  out << "\n\nttd_idx_trip_time_data_:\n";
  for (trip_time_data_idx_t idx{0}; idx < ttd_idx_trip_time_data_.size();
       ++idx) {
    out << "Trip_Time_Data_Idx: " << idx
        << " Start_Time: " << ttd_idx_trip_time_data_[idx].start_timestamp
        << "\n";
    for (auto tsd : ttd_idx_trip_time_data_[idx].seg_data_) {
      out << "Segment: " << tsd.seg_idx << " Progress: " << tsd.progress
          << " Timestamp: " << tsd.timestamp << "\n";
    }
  }
}

std::ostream& operator<<(std::ostream& out,
                         hist_trip_times_storage const& tts) {
  tts.print(out);
  return out;
}

trip_delay_pred delay_prediction_storage::get_or_create_kalman(
    key k,
    unixtime_t start_time,
    uint32_t n_pred,
    uint32_t n_hist,
    hist_trip_times_storage* htts) {
  if (key_trip_delay_.contains(k)) {
    return key_trip_delay_[k];
  }

  trip_delay_pred tdp{};

  tdp.error = 14400;
  tdp.filter_gain = 0.66;
  tdp.gain_loop = 0.33;

  auto const coord_seq_idx = htts->cs_key_coord_seq_[k];

  // find predecessors and historic trips
  vector<trip_time_data_idx_t> predecessors_{};
  std::optional<trip_time_data_idx_t> earliest_pred{};
  vector<trip_time_data_idx_t> hist_trips_{};
  std::optional<trip_time_data_idx_t> earliest_hist{};

  for (auto const ttd_idx : htts->coord_seq_idx_ttd_[coord_seq_idx]) {
    auto& ttd = htts->ttd_idx_trip_time_data_[ttd_idx];

    if (ttd.start_timestamp < start_time) {
      if (!earliest_pred.has_value() ||
          ttd.start_timestamp >
              htts->ttd_idx_trip_time_data_[earliest_pred.value()]
                  .start_timestamp) {
        predecessors_.emplace_back(ttd_idx);
        if (predecessors_.size() == n_pred) {
          earliest_pred = ttd_idx;
        } else if (predecessors_.size() > n_pred) {
          predecessors_.erase(&earliest_pred.value());
        }
      }

      if (((start_time - ttd.start_timestamp) % 10080).count() == 0 &&
          (!earliest_hist.has_value() ||
           ttd.start_timestamp >
               htts->ttd_idx_trip_time_data_[earliest_hist.value()]
                   .start_timestamp)) {
        hist_trips_.emplace_back(ttd_idx);
        if (hist_trips_.size() == n_hist) {
          earliest_hist = ttd_idx;
        } else if (hist_trips_.size() > n_hist) {
          hist_trips_.erase(&earliest_hist.value());
        }
      }
    }
  }
  tdp.predecessors_ = predecessors_;
  tdp.hist_trips_ = hist_trips_;

  // calculation of average historic stop/segment times
  if (!tdp.hist_trips_.empty()) {
    auto const [avg_stop_durations, avg_segment_durations] =
        get_avg_stop_segment_durations(htts, tdp.hist_trips_);
    tdp.hist_avg_stop_durations_ = avg_stop_durations;
    tdp.hist_avg_segment_durations_ = avg_segment_durations;
  }

  key_trip_delay_.emplace(k, tdp);

  return tdp;
}

pair<vector<duration_t>, vector<duration_t>>
delay_prediction_storage::get_avg_stop_segment_durations(
    hist_trip_times_storage* htts, vector<trip_time_data_idx_t> trips) {
  vector<duration_t> avg_stop_durations{};
  vector<duration_t> avg_segment_durations{};

  for (uint32_t i = 0;
       i < htts->ttd_idx_trip_time_data_[trips[0]].stop_durations_.size();
       i++) {
    duration_t sum{0};
    for (auto const hist_trip_idx : trips) {
      sum += htts->ttd_idx_trip_time_data_[hist_trip_idx].stop_durations_[i];
    }
    avg_stop_durations.emplace_back(sum / trips.size());
  }
  for (uint32_t i = 0;
       i < htts->ttd_idx_trip_time_data_[trips[0]].segment_durations_.size();
       i++) {
    duration_t sum{0};
    for (auto const hist_trip_idx : trips) {
      sum += htts->ttd_idx_trip_time_data_[hist_trip_idx].segment_durations_[i];
    }
    avg_segment_durations.emplace_back(sum / trips.size());
  }
  return {avg_stop_durations, avg_segment_durations};
}

duration_t delay_prediction_storage::get_avg_duration(
    vector<duration_t> const& durations) {
  using rep = duration_t::rep;

  if (durations.empty()) {
    return duration_t{0};
  }

  std::int64_t sum = 0;
  for (auto const& d : durations) {
    sum += static_cast<std::int64_t>(d.count());
  }

  auto const n = static_cast<std::int64_t>(durations.size());
  std::int64_t avg = (sum >= 0 ? (sum + n / 2) / n : -(((-sum) + n / 2) / n));

  if (avg > static_cast<std::int64_t>(std::numeric_limits<rep>::max())) {
    avg = std::numeric_limits<rep>::max();
  }

  return duration_t{static_cast<rep>(avg)};
}

void delay_prediction_storage::print(std::ostream& out) const {
  out << "\ncs_key_coord_seq_:\n";
  for (auto const& [key, tdp] : key_trip_delay_) {
    out << "Key: Source: " << key.source_idx << " Trip: " << key.t_idx
        << "\nTrip Delay Predicton: "
           "\nfilter gain: "
        << tdp.filter_gain << "\ngain loop: " << tdp.gain_loop
        << "\nerror: " << tdp.error << "\npredecessors: ";
    if (!tdp.predecessors_.empty()) {
      for (auto const& p : tdp.predecessors_) out << p << ", ";
    } else {
      out << "<none>";
    }
    out << "\nhist trips: ";
    for (auto const hist_trip_idx : tdp.hist_trips_) {
      out << hist_trip_idx << ", ";
    }
    out << "\nhist avg stop durations: ";
    for (auto const stop_dur : tdp.hist_avg_stop_durations_) {
      out << stop_dur.count() << ", ";
    }
    out << "\nhist avg segment durations: ";
    for (auto const seg_dur : tdp.hist_avg_segment_durations_) {
      out << seg_dur.count() << ", ";
    }
    out << "\n\n";
  }
}

std::ostream& operator<<(std::ostream& out,
                         delay_prediction_storage const& dps) {
  dps.print(out);
  return out;
}

}  // namespace nigiri::rt
