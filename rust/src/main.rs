use anyhow::Result;
use chrono::NaiveDate;
use chrono::Utc;
use clap::{Args, Parser, Subcommand};
use std::{net::IpAddr, path::PathBuf};

#[cxx::bridge]
mod nigiri {
    struct LoaderConfig<'a> {
        link_stop_distance: u32,
        default_tz: &'a str,
    }

    unsafe extern "C++" {
        include!("nigiri/rust.h");

        type Timetable;

        fn load_timetable(
            paths: &Vec<String>,
            config: &LoaderConfig,
            start_date: &str,
            num_days: u32,
        ) -> Result<UniquePtr<Timetable>>;

        fn dump_timetable(
            tt: &Timetable,
            path: &str,
        ) -> Result<()>;
    }
}

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

        let tt = nigiri::load_timetable(
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
    timetable: PathBuf,

    #[clap(long, short = 'h')]
    host: IpAddr,

    #[clap(long, short = 'p')]
    port: u16,
}

impl ServeArgs {
    fn exec(&self) -> Result<()> {
        println!("Hello, world!");
        Ok(())
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Prepare(p) => p.exec(),
        Command::Serve(r) => r.exec(),
    }?;

    Ok(())
}
