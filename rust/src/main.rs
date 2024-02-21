use anyhow::Result;
use axum::extract::Query;
use axum::routing::get;
use axum::{debug_handler, extract::State, Router};
use chrono::NaiveDate;
use chrono::Utc;
use clap::{Args, Parser, Subcommand};
use cxx::UniquePtr;
use hyper::StatusCode;
use serde::Deserialize;
use std::collections::HashMap;
use std::net::IpAddr;
use std::net::Ipv4Addr;
use std::sync::Arc;

#[cxx::bridge(namespace = "nigiri::rust")]
mod nigiri {
    struct LoaderConfig<'a> {
        link_stop_distance: u32,
        default_tz: &'a str,
    }

    enum LocationMatchMode {
        Exact,
        OnlyChildren,
        Equivalent,
        Intermodal,
    }

    enum Direction {
        Forward,
        Backward,
    }

    struct Offset {
        target: u32,
        duration: i16,
        idx: i32,
    }

    struct Query {
        search_direction: Direction,
        start_time: i64, // unixtime in seconds
        end_time: i64,   // unixtime in seconds, i64::MIN = earliest arrival query
        start_match_mode: LocationMatchMode,
        dest_match_mode: LocationMatchMode,
        use_start_footpaths: bool,
        start: Vec<Offset>,
        dest: Vec<Offset>,
        max_transfers: u8,
        min_connection_count: u32,
        extend_interval_earlier: bool,
        extend_interval_later: bool,
        profile_idx: u8,
    }

    struct Run {
        transport_idx: u32,
        rt_transport_idx: u32,
        from: u16,
        to: u16,
    }

    enum LegType {
        RunEnterExit,
        Footpath,
        Offset,
    }

    struct Leg {
        from: u32,
        to: u32,
        dep_time: i64,
        arr_time: i64,
        t: LegType,
        run: Run,
        enter: u16,
        exit: u16,
        offset: Offset,
    }

    struct Journey {
        legs: Vec<Leg>,
    }

    struct Station {
        id: String,
        name: String,
    }

    struct Transport<'a> {
        //
        id: str,
    }

    unsafe extern "C++" {
        include!("nigiri/rust/timetable.h");

        type Timetable;

        fn parse_timetables(
            paths: &Vec<String>,
            config: &LoaderConfig,
            start_date: &str,
            num_days: u32,
        ) -> Result<UniquePtr<Timetable>>;

        fn dump_timetable(
            tt: &Timetable,
            path: &str,
        ) -> Result<()>;

        fn load_timetable(path: &str) -> Result<UniquePtr<Timetable>>;

        fn stations(tt: &Timetable) -> Vec<Station>;

        fn<'a> transport(
            tt: &'a Timetable,
            transport_idx: u32,
        ) -> Transport<'a>;
    }
}

unsafe impl Send for nigiri::Timetable {}

unsafe impl Sync for nigiri::Timetable {}

#[derive(Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Prepare(PrepareArgs),
    Serve(ServeArgs),
}

#[derive(Args)]
struct PrepareArgs {
    input: Vec<String>,

    #[clap(long, short = 'o', default_value_t = String::from("tt.bin"))]
    output: String,

    #[clap(long, short = 's', default_value_t = String::from("TODAY"))]
    start_date: String,

    #[clap(long, short = 'n', default_value_t = 256)]
    num_days: u32,

    #[clap(long, short = 'l', default_value_t = 100)]
    link_stop_distance: u32,

    #[clap(long, short = 't', default_value_t = String::from("Europe/Berlin"))]
    default_tz: String,
}

impl PrepareArgs {
    fn exec(&self) -> Result<()> {
        let start_date = if "TODAY" == self.start_date {
            Ok(Utc::now().naive_utc().date())
        } else {
            NaiveDate::parse_from_str(&self.start_date, "%Y-%m-%d")
        }?;
        let start_date = format!("{}", NaiveDate::format(&start_date, "%Y-%m-%d"));

        let tt = nigiri::parse_timetables(
            &self.input,
            &nigiri::LoaderConfig {
                link_stop_distance: self.link_stop_distance,
                default_tz: &self.default_tz,
            },
            &start_date,
            self.num_days,
        );
        nigiri::dump_timetable(&*tt?, &self.output)?;

        Ok(())
    }
}

#[derive(Args)]
struct ServeArgs {
    #[clap(default_value_t = String::from("tt.bin"))]
    timetable: String,

    #[clap(long, short = 'h', default_value_t = IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)))]
    host: IpAddr,

    #[clap(long, short = 'p', default_value_t = 8080)]
    port: u16,
}

#[derive(Clone)]
struct Tags {
    tag_to_idx: HashMap<String, u32>,
    idx_to_tag: Vec<String>,
}

#[derive(Clone)]
struct TimetableInfo {
    tt: Arc<UniquePtr<nigiri::Timetable>>,
    tags: Tags,
}

#[derive(Deserialize)]
struct RoutingQuery {
    from: String,
    to: String,
}

#[debug_handler]
async fn route(
    query: Query<RoutingQuery>,
    State(_tt_info): State<TimetableInfo>,
) -> (StatusCode, String) {
    (StatusCode::OK, "<h1>It works!</h1>".to_string())
}

impl ServeArgs {
    async fn exec(&self) -> Result<()> {
        let tt_info = TimetableInfo {
            tt: Arc::new(nigiri::load_timetable(&self.timetable)?),
            tags: Tags {
                tag_to_idx: HashMap::new(),
                idx_to_tag: Vec::new(),
            },
        };
        let app = Router::new()
            .route("/v1/route", get(route))
            .with_state(tt_info);

        let listener = tokio::net::TcpListener::bind(format!("{}:{}", self.host, self.port))
            .await
            .unwrap();
        axum::serve(listener, app).await.unwrap();

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Prepare(p) => p.exec(),
        Command::Serve(r) => r.exec().await,
    }?;

    Ok(())
}
