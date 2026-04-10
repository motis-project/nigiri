use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use transit_core::TransitResult;

/// Tracks SHA-256 hashes for incremental rebuild detection.
pub struct ImportHashes {
    hashes: HashMap<String, String>,
    path: PathBuf,
}

impl ImportHashes {
    /// Load existing hashes from the meta directory, or create empty.
    pub fn load(data_path: &Path) -> Self {
        let path = data_path.join("meta").join("hashes.json");
        let hashes = if path.exists() {
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            HashMap::new()
        };
        Self { hashes, path }
    }

    /// Save current hashes to disk.
    pub fn save(&self) -> TransitResult<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.hashes)
            .map_err(|e| transit_core::TransitError::Config(e.to_string()))?;
        std::fs::write(&self.path, json)?;
        Ok(())
    }

    /// Check if a component needs rebuilding based on its input files.
    pub fn needs_rebuild(&self, component: &str, inputs: &[&Path]) -> bool {
        let current = hash_inputs(inputs);
        self.hashes.get(component) != Some(&current)
    }

    /// Record the current hash for a component.
    pub fn mark_built(&mut self, component: &str, inputs: &[&Path]) {
        let current = hash_inputs(inputs);
        self.hashes.insert(component.to_string(), current);
    }
}

/// Hash the contents of input files/directories.
fn hash_inputs(paths: &[&Path]) -> String {
    let mut hasher = Sha256::new();
    for path in paths {
        if path.is_file() {
            if let Ok(data) = std::fs::read(path) {
                hasher.update(&data);
            }
        } else if path.is_dir() {
            // Hash directory listing (names + sizes)
            if let Ok(entries) = std::fs::read_dir(path) {
                let mut names: Vec<_> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| {
                        let meta = e.metadata().ok();
                        let size = meta.map(|m| m.len()).unwrap_or(0);
                        format!("{}:{}", e.file_name().to_string_lossy(), size)
                    })
                    .collect();
                names.sort();
                for name in &names {
                    hasher.update(name.as_bytes());
                }
            }
        }
    }
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn hash_detection() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path();

        let mut hashes = ImportHashes::load(data_path);
        assert!(hashes.needs_rebuild("test", &[]));

        hashes.mark_built("test", &[]);
        assert!(!hashes.needs_rebuild("test", &[]));

        // Create a file and check it changes the hash
        let file_path = data_path.join("input.txt");
        let mut f = std::fs::File::create(&file_path).unwrap();
        writeln!(f, "hello").unwrap();

        assert!(hashes.needs_rebuild("test", &[&file_path]));
        hashes.mark_built("test", &[&file_path]);
        assert!(!hashes.needs_rebuild("test", &[&file_path]));
    }

    #[test]
    fn save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let data_path = dir.path();

        let mut hashes = ImportHashes::load(data_path);
        hashes.mark_built("timetable", &[]);
        hashes.save().unwrap();

        let loaded = ImportHashes::load(data_path);
        assert!(!loaded.needs_rebuild("timetable", &[]));
    }
}
