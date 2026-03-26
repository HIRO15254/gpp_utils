use serde::{Serialize, Deserialize};
use std::path::Path;
use std::fs;
use std::io::{Read, Write};

pub fn save_json<T: Serialize>(data: &T, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let json_string = serde_json::to_string_pretty(data)?;
    let mut file = fs::File::create(path)?;
    file.write_all(json_string.as_bytes())?;
    Ok(())
}

pub fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    let mut file = fs::File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let data: T = serde_json::from_str(&contents)?;
    Ok(data)
}

pub fn ensure_dir_exists(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}