# Core Functionality and Main Data Structure of Nigiri

The heart of the Nigiri library is the `nigiri::timetable` struct, defined in `include/nigiri/timetable.h`. This massive data structure serves as a comprehensive, in-memory representation of an entire public transit schedule. Its primary role is to load, store, and provide efficient access to all aspects of transit data, including static schedule information (stops, routes, trips, times, calendars), geographical data, and connections necessary for journey planning.

The `timetable` is designed to handle potentially large and complex datasets, such as those found in GTFS (General Transit Feed Specification) feeds, and make this information readily available for routing algorithms and other transit-related queries. It achieves this by organizing data into various interconnected components, often using dense arrays and lookup tables indexed by specialized strong-typed IDs (e.g., `location_idx_t`, `route_idx_t`, `transport_idx_t`).

## Locations (`timetable::locations_`)

Within the `timetable` struct, a nested struct named `locations_` (of type `timetable::locations`) is responsible for managing all data related to physical places: stops, stations, platforms, points of interest, etc. Each distinct location is assigned a unique `nigiri::location_idx_t`, which serves as its primary internal identifier.

The `locations_` substructure stores various attributes for each location, primarily using `vector_map` and `vecvec` collections indexed by `location_idx_t`:

*   **Identifiers and Names:**
    *   `ids_`: External IDs (e.g., GTFS `stop_id`).
    *   `names_`: Human-readable names (e.g., "Main Street Station").
    *   `descriptions_`: Additional descriptive text.
    *   `location_id_to_idx_`: A hash map to efficiently translate external string IDs (combined with a data source identifier `source_idx_t`) into the internal `location_idx_t`.
*   **Geographical Information:**
    *   `coordinates_`: Stores `geo::latlng` (latitude and longitude) for each location.
    *   `rtree_`: An R-tree is built using these coordinates to enable fast spatial queries, such as finding nearby locations.
*   **Hierarchical and Grouping Information:**
    *   `parents_`: Stores the `location_idx_t` of a parent station if the location is a child (e.g., a platform or a specific stop place within a larger station complex).
    *   `children_`: A multimap storing `location_idx_t` of child locations for a parent station.
    *   `equivalences_`: A multimap to group multiple `location_idx_t` that refer to the same physical point (e.g., different stop IDs from different agencies for the same bus stop).
*   **Transit Properties:**
    *   `types_`: Specifies the `nigiri::location_type` (e.g., station, track).
    *   `location_timezones_`: Associates each location with a `nigiri::timezone_idx_t`, which points to timezone information (critical for correct time interpretation).
    *   `transfer_time_`: The default time (in minutes) required to transfer to/from this location.
*   **Footpaths:**
    *   `footpaths_out_` and `footpaths_in_`: Store precomputed footpaths (walking connections) to and from other locations, respectively. These are typically generated based on proximity and are essential for routing that involves walking between stops. These are stored per profile (e.g., regular walking, wheelchair).

This comprehensive storage allows Nigiri to quickly access all relevant details for any given location using its `location_idx_t`.

## Routes

Routes in Nigiri define the path a transit vehicle takes, represented as an ordered sequence of stops. Each unique route pattern is assigned a `nigiri::route_idx_t`. The `timetable` struct stores route information directly, not in a separate substructure like locations.

Key data members for routes include:

*   **`route_location_seq_`**: This is a `vecvec<route_idx_t, stop::value_type>`. For each `route_idx_t`, it stores a vector of `stop::value_type` elements. A `stop::value_type` is essentially a `location_idx_t` combined with flags indicating whether boarding (`in_allowed_`) and alighting (`out_allowed_`) are permitted at that stop on this route (and similarly for wheelchair accessibility). This defines the exact sequence of stops and their properties for a given route.
*   **`route_clasz_`**: A `vector_map<route_idx_t, clasz>` that stores the primary `nigiri::clasz` (e.g., `kBus`, `kTrain`, `kTram`) for each route. This indicates the general type of vehicle serving the route.
*   **`route_section_clasz_`**: A `vecvec<route_idx_t, clasz>` that can store different `clasz` values for different sections of the same route, if the vehicle type changes mid-route (though less common).
*   **Accessibility Information:**
    *   `route_bikes_allowed_`: A bitvector indicating if bicycles are generally allowed on the entire route or on parts of it.
    *   `route_bikes_allowed_per_section_`: If bikes are allowed only on specific sections, this `vecvec` stores boolean flags for each section of the route.
    *   Similar structures (`route_cars_allowed_`, `route_cars_allowed_per_section_`) exist for car transport, likely for services like car-carrying trains.
*   **`location_routes_`**: A `vecvec<location_idx_t, route_idx_t>` which provides a reverse mapping: for each location, it lists all routes (`route_idx_t`) that pass through it. This is useful for finding which routes serve a particular stop.

Routes form a foundational layer upon which specific trips and transports are built. They define the "where" and "how" (vehicle type) of a transit service.

## Services / Traffic Days (`bitfield`)

To specify when a particular transit service operates, Nigiri uses `nigiri::bitfield` objects. A `bitfield` is essentially a bitset (defaulting to `kMaxDays = 512` bits) where each bit corresponds to a specific day within the timetable's active period. If a bit is set, the service operates on that day; if not, it doesn't. This allows for flexible representation of service patterns like "weekdays only," "weekends," or specific calendar dates.

The `timetable` stores these as follows:

*   **`bitfields_`**: This `vector_map<bitfield_idx_t, bitfield>` stores all the unique `bitfield` patterns found in the transit data. Each unique pattern is assigned a `nigiri::bitfield_idx_t`.
*   **`transport_traffic_days_`**: This `vector_map<transport_idx_t, bitfield_idx_t>` links each `transport` (explained in the next section) to a `bitfield_idx_t`. This effectively defines the operating calendar for that specific transport.

By looking up the `bitfield_idx_t` for a transport and then consulting the corresponding `bitfield` in `bitfields_`, Nigiri can quickly determine if that transport is running on any given day within the timetable's range. The `timetable` also defines a `date_range_` (the overall validity period) and helper functions like `day_idx(date::sys_days)` to convert actual dates into indices for these bitfields. The constant `kTimetableOffset` is used to handle days outside the nominal range that might occur due to timezone conversions or long trips.

## Transports: The Core Operational Units

The `nigiri::transport` entity is arguably the most crucial concept for understanding how Nigiri represents actual vehicle movements. Each `transport` is assigned a unique `nigiri::transport_idx_t` and represents a single, scheduled run of a vehicle along a specific route on the days defined by its service calendar.

Here's how transports tie together various pieces of information:

*   **`transport_route_`**: A `vector_map<transport_idx_t, route_idx_t>` that links each transport to the `nigiri::route_idx_t` it follows. This defines the sequence of stops for the transport.
*   **`transport_traffic_days_`**: As explained previously, this `vector_map<transport_idx_t, bitfield_idx_t>` links the transport to a `bitfield_idx_t`, specifying its operating days.
*   **Event Times (Arrivals and Departures):**
    *   The actual arrival and departure times for each stop on a transport's route are stored in a large, flat vector called **`route_stop_times_`**. This vector contains `nigiri::delta` objects. A `delta` is a compact representation of time, storing days and minutes-after-midnight, allowing times to exceed 24:00 (e.g., 25:30 for 1:30 AM the next day).
    *   To find the times for a specific transport at a specific stop, Nigiri uses **`route_transport_ranges_`** (`vector_map<route_idx_t, interval<transport_idx_t>>`). This indicates, for each route, which transports belong to it.
    *   Additionally, **`route_stop_time_ranges_`** (`vector_map<route_idx_t, interval<std::uint32_t>>`) provides an index range into the `route_stop_times_` vector for the event times of the first transport associated with that route.
    *   The timetable provides helper methods like `event_mam(transport_idx_t t, stop_idx_t const stop_idx, event_type const ev_type)` to retrieve the specific `delta` (time) for a given transport at a particular stop (specified by its index `stop_idx_t` within the route's stop sequence) for either an arrival or departure event. The time is relative to the start of the service day.
*   **`transport_first_dep_offset_`**: A `vector_map<transport_idx_t, duration_t>` that stores the offset between the time stored internally and the time given in the original GTFS timetable. This is important for matching GTFS-RT (real-time) updates, as GTFS stop times can have departure/arrival times like "25:00:00". Nigiri normalizes these times (e.g., to "01:00:00" on day D+1) and stores the offset.

In essence, a `transport` brings together the "what" (vehicle/service attributes via its route), "where" (the sequence of stops from its route), "when" (the operating days from its bitfield), and "at what time" (the schedule of arrivals and departures from `route_stop_times_`). Routing algorithms primarily work with these transport entities to construct journeys.

## Mapping GTFS Trips to Transports

While `transport` entities are the core operational units within Nigiri, data often originates from formats like GTFS, which has its own concept of "trips" (usually from `trips.txt` and `stop_times.txt`). Nigiri provides mechanisms to map these external trip identifiers to its internal transport structure.

*   **`trip_id_strings_` and `trip_id_src_`**: These store the raw string identifiers of trips from the input data (e.g., the `trip_id` from GTFS) and their source dataset (`source_idx_t`). Each unique string ID from a source gets a `trip_id_idx_t`.
*   **`trip_ids_`**: This is a `mutable_fws_multimap<trip_idx_t, trip_id_idx_t>`. It links an internal `trip_idx_t` to one or more external `trip_id_idx_t`. This allows Nigiri to consolidate multiple external trip representations that refer to the same underlying service into a single internal `trip_idx_t`.
*   **`merged_trips_`**: A `vecvec<merged_trips_idx_t, trip_idx_t>` which groups one or more `trip_idx_t` together. This is used, for example, when several distinct GTFS trips are found to be part of the same block or represent the exact same journey and can be merged into a single logical service run, identified by a `merged_trips_idx_t`.
*   **`transport_to_trip_section_`**: This `vecvec<transport_idx_t, merged_trips_idx_t>` is key for linking the operational `transport` back to the original trip concepts. Each transport is associated with one or more `merged_trips_idx_t`. This structure can handle complex scenarios, such as:
    *   A single transport covering exactly one GTFS trip.
    *   A transport representing a service that was defined by merging several GTFS trips.
    *   A transport that might only cover a section of a longer conceptual trip defined in the input.
*   **`trip_display_names_`**: Stores human-readable names for trips, often derived from GTFS `trip_headsign` or `route_short_name`, associated with `trip_idx_t`.
*   **`trip_debug_`**: Contains debugging information, linking `trip_idx_t` back to source file names and line numbers from the input GTFS, aiding in traceability.

This layered approach allows Nigiri to normalize and consolidate varied trip representations from input data into its more uniform `transport`-based model, while still retaining links back to the original trip identifiers for purposes like real-time updates (GTFS-RT messages often refer to GTFS `trip_id`s).

## Other Important Components

Besides the core structures for locations, routes, services, and transports, the `nigiri::timetable` manages several other crucial pieces of information:

*   **Time Representation:**
    *   Nigiri uses specialized types for time and duration, defined in `include/nigiri/types.h`.
    *   `unixtime_t`: Represents absolute points in time, stored as a `std::chrono::sys_time` with 32-bit minutes precision. Used for date ranges and specific timestamps.
    *   `duration_t`: Represents durations, stored as `std::chrono::duration` with 16-bit minutes precision. Used for transfer times, journey durations, etc.
    *   `delta`: As mentioned earlier, this compact struct (5 bits for days, 11 bits for minutes-after-midnight) is crucial for representing event times relative to a service day, allowing schedules to naturally span past midnight.
    *   `timezone` and `tz_offsets`: Manage timezone conversions, critical for correctly interpreting local times from various sources and converting them to a consistent internal representation (usually UTC-based for calculations).
*   **Providers (Agencies):**
    *   `providers_`: A `vector_map<provider_idx_t, provider>` stores information about transit agencies (e.g., name, URL, timezone), each identified by a `provider_idx_t`.
*   **Fares:**
    *   `fares_`: A `vector_map<source_idx_t, fares>` can store fare information, often loaded from GTFS `fare_attributes.txt` and `fare_rules.txt`. The exact structure of `nigiri::fares` would detail how fare zones, ticket types, and costs are represented.
*   **Attributes:**
    *   `attributes_` and `attribute_combinations_`: Store additional descriptive attributes for trips or sections (e.g., "wheelchair accessible," "bicycle carriage available"), often from GTFS `trips.txt` or via custom extensions.
*   **Flexible Transport (`flex_` prefixed members):**
    *   Nigiri includes support for flexible or demand-responsive transport services. This involves structures like:
        *   `flex_area_bbox_`, `flex_area_locations_`, `flex_area_rtree_`: Defining geographical areas for flex services.
        *   `location_group_locations_`: Grouping locations for flex services.
        *   `flex_transport_traffic_days_`, `flex_transport_trip_`: Defining when and which trips operate as flexible services.
        *   `booking_rules_`: Storing rules related to booking flexible services.
*   **String Storage:**
    *   `strings_`: A `nigiri::string_store` is used to efficiently store and manage all unique string literals (like names, IDs, URLs) to reduce memory footprint by interning them. Many data structures store `string_idx_t` instead of raw strings.
*   **Lower Bound Graphs for Routing:**
    *   `fwd_search_lb_graph_` and `bwd_search_lb_graph_`: These arrays (one per profile) of `vecvec<location_idx_t, footpath>` store precomputed graphs representing minimum travel times (lower bounds) between locations. These are used by routing algorithms like Raptor to speed up searches.

All these components contribute to making the `nigiri::timetable` a self-contained and highly optimized data store for public transit information, tailored for efficient querying and routing.
