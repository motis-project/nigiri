#include "nigiri/loader/gtfs/transfer.h"

#include <algorithm>
#include <tuple>

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"

#include "nigiri/logging.h"
#include "utl/pipes/remove_if.h"

namespace nigiri::loader::gtfs {

hash_map<stop_pair, transfer> read_transfers(stop_map const& stops,
                                             std::string_view file_content) {
  nigiri::scoped_timer timer{"read transfers"};

  struct csv_transfer {
    utl::csv_col<utl::cstr, UTL_NAME("from_stop_id")> from_stop_id_;
    utl::csv_col<utl::cstr, UTL_NAME("to_stop_id")> to_stop_id_;
    utl::csv_col<int, UTL_NAME("transfer_type")> transfer_type_;
    utl::csv_col<int, UTL_NAME("min_transfer_time")> min_transfer_time_;
  };

  std::map<stop_pair, transfer> transfers;

  if (file_content.empty()) {
    return {};
  }

  return utl::line_range{utl::buf_reader{file_content}}  //
         | utl::csv<csv_transfer>()  //
         |
         utl::transform([&](csv_transfer const& t)
                            -> std::optional<cista::pair<stop_pair, transfer>> {
           auto const from_stop_it = stops.find(t.from_stop_id_->view());
           if (from_stop_it == end(stops)) {
             log(log_lvl::error, "loader.gtfs.transfers", "stop {} not found\n",
                 t.from_stop_id_->view());
             return std::nullopt;
           }

           auto const to_stop_it = stops.find(t.to_stop_id_->view());
           if (from_stop_it == end(stops)) {
             log(log_lvl::error, "loader.gtfs.transfers", "stop {} not found\n",
                 t.to_stop_id_->view());
             return std::nullopt;
           }

           auto const k =
               std::pair{from_stop_it->second.get(), to_stop_it->second.get()};
           auto const v = transfer{
               .type_ = static_cast<transfer::type>(*t.transfer_type_),
               .minutes_ = u8_minutes{
                   static_cast<std::uint8_t>(*t.min_transfer_time_ / 60)}};

           return std::make_optional<cista::pair<stop_pair, transfer>>(k, v);
         })  //
         | utl::remove_if([](auto&& opt) { return !opt.has_value(); })  //
         | utl::transform([](auto&& opt) { return *opt; })  //
         | utl::to<hash_map<stop_pair, transfer>>();
}

}  // namespace nigiri::loader::gtfs
