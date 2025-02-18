#include "nigiri/loader/assistance.h"

#include "geo/point_rtree.h"

#include "utl/logging.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

namespace nigiri::loader {

assistance_times read_assistance(std::string_view file_content) {
  struct assistance {
    utl::csv_col<utl::cstr, UTL_NAME("name")> name_;
    utl::csv_col<double, UTL_NAME("lat")> lat_;
    utl::csv_col<double, UTL_NAME("lng")> lng_;
    utl::csv_col<utl::cstr, UTL_NAME("time")> time_;
  };

  auto a = assistance_times{};
  utl::line_range{utl::make_buf_reader(file_content)}  //
      | utl::csv<assistance>()  //
      |
      utl::for_each([&](assistance const& x) {
        a.names_.emplace_back(x.name_->trim().view());
        a.pos_.emplace_back(*x.lat_, *x.lng_);
        try {
          a.rules_.emplace_back(oh::parse(x.time_->trim().view()));
        } catch (std::exception const& e) {
          utl::log_error("loader.assistance", "bad assistance time \"{}\": {}",
                         x.time_->view(), e.what());
        }
      });
  a.rtree_ = geo::make_point_rtree(a.pos_);
  return a;
}

}  // namespace nigiri::loader