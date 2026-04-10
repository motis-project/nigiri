#[derive(Debug, thiserror::Error)]
pub enum TransitError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid ID format: {0}")]
    InvalidId(String),

    #[error("nigiri error: {0}")]
    Nigiri(String),

    #[error("osr error: {0}")]
    Osr(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("yaml parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

pub type TransitResult<T> = Result<T, TransitError>;
