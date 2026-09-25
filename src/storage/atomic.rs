use crate::error::{Result, UnsupportedSchema};
use anyhow::Context;
use fs2::FileExt;
use serde::Serialize;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};

pub struct WriterLock(File);
impl WriterLock {
    pub fn acquire(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join(".writer.lock"))?;
        file.try_lock_exclusive()
            .context("another writer is using this data root")?;
        Ok(Self(file))
    }
}
impl Drop for WriterLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}
pub fn writer_active(root: &Path) -> Result<bool> {
    let path = root.join(".writer.lock");
    if !path.exists() {
        return Ok(false);
    }
    let f = match OpenOptions::new().read(true).write(true).open(path) {
        Ok(f) => f,
        Err(e) if lock_contended(&e) => return Ok(true),
        Err(e) => return Err(e.into()),
    };
    match f.try_lock_exclusive() {
        Ok(()) => {
            FileExt::unlock(&f)?;
            Ok(false)
        }
        Err(e) if lock_contended(&e) => Ok(true),
        Err(e) => Err(e.into()),
    }
}
pub fn write_bytes(path: &Path, bytes: &[u8], overwrite: bool) -> Result<()> {
    let parent = path.parent().context("output path has no parent")?;
    std::fs::create_dir_all(parent)?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    tmp.write_all(bytes)?;
    tmp.as_file().sync_all()?;
    if overwrite {
        tmp.persist(path).map_err(|e| e.error)?;
    } else {
        tmp.persist_noclobber(path).map_err(|e| e.error)?;
    }
    #[cfg(unix)]
    File::open(parent)?.sync_all()?;
    Ok(())
}
/// Atomically writes pretty-printed JSON followed by a newline. Experiment,
/// graph and export metadata files use this human-readable form.
pub fn write_json<T: Serialize>(path: &Path, data: &T, overwrite: bool) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(data)?;
    bytes.push(b'\n');
    write_bytes(path, &bytes, overwrite)
}
/// Atomically writes compact JSON (a single line without insignificant
/// whitespace) followed by a newline. Per-run result files and incomplete
/// markers use this form; [`read_json`] reads both forms.
pub fn write_json_compact<T: Serialize>(path: &Path, data: &T, overwrite: bool) -> Result<()> {
    let mut bytes = serde_json::to_vec(data)?;
    bytes.push(b'\n');
    write_bytes(path, &bytes, overwrite)
}
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid JSON: {}", path.display()))?;
    if let Some(version) = value.get("schema_version")
        && version.as_u64() != Some(1)
    {
        return Err(UnsupportedSchema { path: path.into() }.into());
    }
    serde_json::from_value(value).with_context(|| format!("invalid data: {}", path.display()))
}

fn lock_contended(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::WouldBlock
        || (cfg!(windows) && error.raw_os_error() == Some(33))
}
