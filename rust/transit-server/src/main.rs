mod app_data;
mod gql_types;
mod schema;

use std::net::SocketAddr;
use std::path::PathBuf;

use async_graphql::http::{playground_source, GraphQLPlaygroundConfig};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use axum::{
    extract::State,
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use tower_http::cors::CorsLayer;
use tracing_subscriber::EnvFilter;

use app_data::AppData;
use schema::{build_schema, AppSchema};
use transit_core::Config;

async fn graphql_handler(State(schema): State<AppSchema>, req: GraphQLRequest) -> GraphQLResponse {
    schema.execute(req.into_inner()).await.into()
}

async fn graphql_playground() -> impl IntoResponse {
    Html(playground_source(GraphQLPlaygroundConfig::new("/graphql")))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.yaml".to_string());

    let config = if std::path::Path::new(&config_path).exists() {
        Config::load(&config_path)?
    } else {
        tracing::warn!("Config file not found at {config_path}, using defaults");
        Config::default()
    };

    let host = config.server.host.clone();
    let port = config.server.port;
    let data_path = PathBuf::from(".");

    // Run import pipeline if timetable is configured
    let data = if config.timetable.is_some() {
        match transit_import::run_import(&config, &data_path) {
            Ok(import_result) => {
                tracing::info!("Import pipeline complete");
                AppData::from_import(config, data_path, import_result)
            }
            Err(e) => {
                tracing::error!("Import failed: {e}");
                tracing::warn!("Starting server without timetable data");
                AppData::empty(config, data_path)
            }
        }
    } else {
        tracing::info!("No timetable configured, starting with health-check only");
        AppData::empty(config, data_path)
    };

    let schema = build_schema(data);

    let app = Router::new()
        .route("/graphql", get(graphql_playground).post(graphql_handler))
        .layer(CorsLayer::permissive())
        .with_state(schema);

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!("GraphQL playground at http://{addr}/graphql");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
