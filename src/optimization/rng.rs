use rand_mt::Mt19937GenRand64;
use sha2::{Digest, Sha256};

pub fn derive_seed(parts: &[&[u8]]) -> u64 {
    let mut hash = Sha256::new();
    hash.update(b"gpp-utils-rng-v1\0");
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    u64::from_le_bytes(hash.finalize()[..8].try_into().expect("eight bytes"))
}

pub fn rng_for(parts: &[&[u8]]) -> Mt19937GenRand64 {
    Mt19937GenRand64::new(derive_seed(parts))
}
