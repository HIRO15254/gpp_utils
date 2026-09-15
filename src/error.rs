pub use anyhow::{Error, Result};

#[derive(Debug)]
/// A persisted schema version this build cannot interpret safely.
/// Storage matches this type through anyhow contexts to prohibit automatic
/// corruption recovery; changing its human-readable message is safe.
pub struct UnsupportedSchema {
    pub path: std::path::PathBuf,
}

impl std::fmt::Display for UnsupportedSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unsupported schema version in {}", self.path.display())
    }
}
impl std::error::Error for UnsupportedSchema {}
