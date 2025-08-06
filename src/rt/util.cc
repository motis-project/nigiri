#include "nigiri/rt/util.h"

#include "google/protobuf/util/json_util.h"

#include "utl/verify.h"

#include "gtfsrt/gtfs-realtime.pb.h"

namespace nigiri::rt {

std::string json_to_protobuf(std::string_view json) {
  transit_realtime::FeedMessage msg;
  auto const status = google::protobuf::util::JsonStringToMessage(json, &msg);
  utl::verify(status.ok(), "json_to_protobuf: {}", status.message());
  return msg.SerializeAsString();
}

std::string protobuf_to_json(std::string_view protobuf) {
  transit_realtime::FeedMessage msg;
  auto const success =
      msg.ParseFromArray(reinterpret_cast<void const*>(protobuf.data()),
                         static_cast<int>(protobuf.size()));
  utl::verify(success, "json_to_protobuf: read protobuf FeedMessage failed");

  auto json = std::string{};
  auto const status = google::protobuf::util::MessageToJsonString(
      msg, &json, {.add_whitespace = true});
  utl::verify(status.ok(), "protobuf_to_json: {}", status.message());
  return json;
}

}  // namespace nigiri::rt