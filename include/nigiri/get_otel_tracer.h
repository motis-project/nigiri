#pragma once

#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/tracer.h"

#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"

namespace nigiri {

inline opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer>
get_otel_tracer() {
  return opentelemetry::trace::Provider::GetTracerProvider()->GetTracer(
      "nigiri");
}

}  // namespace nigiri
