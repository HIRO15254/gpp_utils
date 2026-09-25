//! Compact JSON representation of a run's partition pool.
//!
//! In memory a pool is `Vec<Vec<bool>>` with one value per vertex; `true` is
//! group A. JSON stores the pool as
//!
//! ```json
//! {"length": 10, "hex": ["0103"]}
//! ```
//!
//! `length` is the vertex count shared by every partition, and `0` for an empty
//! pool. Each `hex` string packs one partition LSB first: vertex `v` is bit
//! `v % 8` of byte `v / 8`, with `true` as `1`. Every byte is two lowercase
//! hexadecimal digits, so a string has exactly `2 * ceil(length / 8)` digits
//! and the unused high bits of the last byte are zero. Decoding rejects every
//! other spelling of the pool object itself. Stored files are read through a
//! JSON value first, where a duplicated key keeps its last occurrence, so this
//! strictness applies to the object that survives that step. Python decodes a string
//! with `np.unpackbits(np.frombuffer(bytes.fromhex(s), np.uint8),
//! bitorder="little")[:length]`.
use serde::{
    Deserializer, Serialize, Serializer,
    de::{self, MapAccess, Visitor},
    ser::{self, SerializeMap},
};
use std::fmt;

const DIGITS: &[u8; 16] = b"0123456789abcdef";
const FIELDS: &[&str] = &["length", "hex"];

/// Serializes a pool whose partitions all have the same length.
pub(crate) fn serialize<S: Serializer>(
    partitions: &[Vec<bool>],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let length = partitions.first().map_or(0, Vec::len);
    if let Some(other) = partitions.iter().find(|p| p.len() != length) {
        return Err(ser::Error::custom(format!(
            "partitions must share one length, found {length} and {}",
            other.len()
        )));
    }
    let mut map = serializer.serialize_map(Some(FIELDS.len()))?;
    map.serialize_entry("length", &length)?;
    map.serialize_entry("hex", &HexList(partitions))?;
    map.end()
}

/// Deserializes and strictly validates the compact pool representation.
pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<Vec<bool>>, D::Error> {
    deserializer.deserialize_map(PoolVisitor)
}

struct HexList<'a>(&'a [Vec<bool>]);
impl Serialize for HexList<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(|partition| encode(partition)))
    }
}

struct PoolVisitor;
impl<'de> Visitor<'de> for PoolVisitor {
    type Value = Vec<Vec<bool>>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a partition pool object with `length` and `hex`")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut length: Option<usize> = None;
        let mut hex: Option<Vec<String>> = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "length" if length.is_some() => return Err(de::Error::duplicate_field("length")),
                "length" => length = Some(map.next_value()?),
                "hex" if hex.is_some() => return Err(de::Error::duplicate_field("hex")),
                "hex" => hex = Some(map.next_value()?),
                other => return Err(de::Error::unknown_field(other, FIELDS)),
            }
        }
        let length = length.ok_or_else(|| de::Error::missing_field("length"))?;
        let hex = hex.ok_or_else(|| de::Error::missing_field("hex"))?;
        decode_pool(length, &hex).map_err(de::Error::custom)
    }
}

fn encode(partition: &[bool]) -> String {
    let mut hex = String::with_capacity(partition.len().div_ceil(8) * 2);
    for chunk in partition.chunks(8) {
        let byte = chunk
            .iter()
            .enumerate()
            .fold(0u8, |byte, (bit, &value)| byte | (u8::from(value) << bit));
        hex.push(char::from(DIGITS[usize::from(byte >> 4)]));
        hex.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    hex
}

fn decode_pool(length: usize, hex: &[String]) -> Result<Vec<Vec<bool>>, String> {
    if hex.is_empty() && length != 0 {
        return Err(format!(
            "invalid partitions: an empty pool must have length 0, found {length}"
        ));
    }
    hex.iter()
        .enumerate()
        .map(|(index, text)| {
            decode(text, length).map_err(|e| format!("invalid partitions: hex[{index}]: {e}"))
        })
        .collect()
}

fn decode(hex: &str, length: usize) -> Result<Vec<bool>, String> {
    for (offset, byte) in hex.bytes().enumerate() {
        match byte {
            b'0'..=b'9' | b'a'..=b'f' => {}
            b'A'..=b'F' => {
                return Err(format!(
                    "uppercase hexadecimal digit {:?} at offset {offset}; digits must be lowercase",
                    char::from(byte)
                ));
            }
            _ => return Err(format!("non-hexadecimal character at offset {offset}")),
        }
    }
    let expected = length.div_ceil(8) * 2;
    if hex.len() != expected {
        return Err(format!(
            "expected {expected} hexadecimal digits for length {length}, found {}",
            hex.len()
        ));
    }
    let mut partition = Vec::with_capacity(length);
    for (index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let byte = (digit_value(pair[0]) << 4) | digit_value(pair[1]);
        let bits = (length - 8 * index).min(8);
        if bits < 8 && byte >> bits != 0 {
            return Err(format!("non-zero padding bits in byte {index}"));
        }
        partition.extend((0..bits).map(|bit| (byte >> bit) & 1 == 1));
    }
    Ok(partition)
}

/// Value of a digit already checked to be `0-9` or `a-f`.
fn digit_value(digit: u8) -> u8 {
    match digit {
        b'0'..=b'9' => digit - b'0',
        _ => digit - b'a' + 10,
    }
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};
    use rand::Rng;
    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Pool {
        #[serde(with = "crate::experiment::partition_codec")]
        partitions: Vec<Vec<bool>>,
    }

    fn to_json(partitions: Vec<Vec<bool>>) -> Value {
        serde_json::to_value(Pool { partitions }).unwrap()["partitions"].clone()
    }
    fn from_json(value: Value) -> Result<Vec<Vec<bool>>, String> {
        serde_json::from_value::<Pool>(json!({ "partitions": value }))
            .map(|pool| pool.partitions)
            .map_err(|e| e.to_string())
    }
    fn rejected(value: Value, message: &str) {
        let error = from_json(value.clone()).expect_err(&format!("accepted {value}"));
        assert!(error.contains(message), "{value}: {error}");
    }
    /// Independent reference packing: one byte per eight vertices, LSB first.
    fn reference_hex(partition: &[bool]) -> String {
        (0..partition.len().div_ceil(8))
            .map(|byte| {
                let value = (0..8)
                    .filter(|bit| partition.get(byte * 8 + bit) == Some(&true))
                    .map(|bit| 1u8 << bit)
                    .sum::<u8>();
                format!("{value:02x}")
            })
            .collect()
    }
    fn random_pool(rng: &mut impl Rng, length: usize, count: usize) -> Vec<Vec<bool>> {
        (0..count)
            .map(|_| (0..length).map(|_| rng.r#gen::<bool>()).collect())
            .collect()
    }

    #[test]
    fn hand_checked_examples_use_lsb_first_lowercase_bytes() {
        let ten = vec![
            true, false, false, false, false, false, false, false, true, true,
        ];
        assert_eq!(encode(&ten), "0103");
        assert_eq!(
            to_json(vec![ten.clone()]),
            json!({"length": 10, "hex": ["0103"]})
        );
        assert_eq!(
            from_json(json!({"length": 10, "hex": ["0103"]})),
            Ok(vec![ten])
        );
        // The documented four-vertex cycle example.
        assert_eq!(
            to_json(vec![
                vec![true, false, true, false],
                vec![true, true, false, false]
            ]),
            json!({"length": 4, "hex": ["05", "03"]})
        );
        assert_eq!(encode(&[false; 8]), "00");
        assert_eq!(encode(&[true; 8]), "ff");
        assert_eq!(encode(&[true; 9]), "ff01");
        let mut high = vec![false; 16];
        high[7] = true;
        high[12] = true;
        assert_eq!(encode(&high), "8010");
        let mut all_digits = vec![false; 64];
        for (vertex, value) in all_digits.iter_mut().enumerate() {
            *value = (0x_fedc_ba98_7654_3210_u64 >> vertex) & 1 == 1;
        }
        assert_eq!(encode(&all_digits), "1032547698badcfe");
    }

    #[test]
    fn round_trip_matches_independent_packing_for_all_lengths() {
        let mut rng = crate::optimization::rng_for(&[b"partition-codec-round-trip"]);
        let lengths = (0..=20).chain([63, 64, 65, 123, 124, 500, 501, 1003]);
        for length in lengths {
            for count in [1, 2, 7] {
                let mut pool = random_pool(&mut rng, length, count);
                if length > 0 {
                    pool.push(vec![true; length]);
                    pool.push(vec![false; length]);
                }
                let value = to_json(pool.clone());
                assert_eq!(value["length"], json!(length));
                let hex = value["hex"].as_array().unwrap();
                assert_eq!(hex.len(), pool.len());
                for (text, partition) in hex.iter().zip(&pool) {
                    let text = text.as_str().unwrap();
                    assert_eq!(text, reference_hex(partition), "length {length}");
                    assert_eq!(text.len(), 2 * length.div_ceil(8));
                    assert_eq!(decode(text, length).unwrap(), *partition);
                }
                assert_eq!(from_json(value).unwrap(), pool, "length {length}");
            }
        }
    }

    #[test]
    fn exhaustive_small_partitions_round_trip() {
        for length in 1..=12usize {
            for mask in 0..(1u32 << length) {
                let partition: Vec<bool> = (0..length).map(|v| (mask >> v) & 1 == 1).collect();
                let text = encode(&partition);
                assert_eq!(text, reference_hex(&partition));
                assert_eq!(decode(&text, length).unwrap(), partition);
            }
        }
    }

    #[test]
    fn empty_pools_and_zero_length_partitions_are_canonical() {
        assert_eq!(to_json(vec![]), json!({"length": 0, "hex": []}));
        assert_eq!(from_json(json!({"length": 0, "hex": []})), Ok(vec![]));
        assert_eq!(to_json(vec![vec![]]), json!({"length": 0, "hex": [""]}));
        assert_eq!(
            from_json(json!({"length": 0, "hex": [""]})),
            Ok(vec![vec![]])
        );
        rejected(
            json!({"length": 3, "hex": []}),
            "an empty pool must have length 0",
        );
    }

    #[test]
    fn serialization_rejects_partitions_of_different_lengths() {
        let error = serde_json::to_value(Pool {
            partitions: vec![vec![true, false], vec![true]],
        })
        .unwrap_err();
        assert!(error.to_string().contains("share one length"), "{error}");
        let error = serde_json::to_vec(&Pool {
            partitions: vec![vec![true; 9], vec![true; 9], vec![true; 16]],
        })
        .unwrap_err();
        assert!(error.to_string().contains("found 9 and 16"), "{error}");
    }

    #[test]
    fn decoding_rejects_wrong_string_lengths() {
        for (length, text) in [
            (10, "01"),
            (10, "010300"),
            (10, "010"),
            (8, ""),
            (0, "00"),
            (1, "0"),
            (17, "ffff01ff"),
        ] {
            rejected(
                json!({"length": length, "hex": [text]}),
                "hexadecimal digits for length",
            );
        }
        rejected(
            json!({"length": 10, "hex": ["0103", "01"]}),
            "hex[1]: expected 4 hexadecimal digits for length 10, found 2",
        );
    }

    #[test]
    fn decoding_rejects_non_hex_uppercase_and_padding() {
        for text in ["0g03", "01 3", "-103", "0x03", "01\u{e9}", "\u{ff10}3"] {
            rejected(
                json!({"length": 10, "hex": [text]}),
                "non-hexadecimal character",
            );
        }
        for text in ["0A03", "01B3", "010F", "FF03"] {
            rejected(json!({"length": 10, "hex": [text]}), "uppercase");
        }
        // Length 10 leaves six unused high bits in the second byte.
        for text in ["0107", "0104", "0183", "01ff"] {
            rejected(
                json!({"length": 10, "hex": [text]}),
                "non-zero padding bits in byte 1",
            );
        }
        rejected(json!({"length": 1, "hex": ["02"]}), "padding");
        rejected(json!({"length": 7, "hex": ["80"]}), "padding");
        assert_eq!(
            from_json(json!({"length": 7, "hex": ["7f"]})),
            Ok(vec![vec![true; 7]])
        );
    }

    #[test]
    fn decoding_rejects_missing_extra_duplicate_and_mistyped_fields() {
        rejected(json!({"hex": ["0103"]}), "missing field `length`");
        rejected(json!({"length": 10}), "missing field `hex`");
        rejected(
            json!({"length": 10, "hex": ["0103"], "bits": []}),
            "unknown field `bits`",
        );
        rejected(json!({}), "missing field");
        rejected(json!({"length": 10, "hex": [259]}), "invalid type");
        rejected(json!({"length": 10, "hex": [null]}), "invalid type");
        rejected(json!({"length": 10, "hex": [["01", "03"]]}), "invalid type");
        rejected(json!({"length": 10, "hex": "0103"}), "invalid type");
        rejected(json!({"length": 10.0, "hex": ["0103"]}), "invalid type");
        rejected(json!({"length": -10, "hex": ["0103"]}), "invalid");
        rejected(json!({"length": "10", "hex": ["0103"]}), "invalid type");
        // Neither the historical bool arrays nor a positional sequence decode.
        rejected(json!([[true, false], [false, true]]), "invalid type");
        rejected(json!([10, ["0103"]]), "invalid type");
        rejected(json!("0103"), "invalid type");
        rejected(json!(null), "invalid type");
    }

    #[test]
    fn duplicate_keys_are_rejected_by_the_streaming_parser() {
        let error = serde_json::from_str::<Pool>(
            r#"{"partitions":{"length":10,"hex":["0103"],"length":10}}"#,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("duplicate field `length`"),
            "{error}"
        );
        let error = serde_json::from_str::<Pool>(
            r#"{"partitions":{"hex":["0103"],"length":10,"hex":["0103"]}}"#,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("duplicate field `hex`"),
            "{error}"
        );
    }

    #[test]
    fn key_order_does_not_matter_for_decoding() {
        let pool: Pool =
            serde_json::from_str(r#"{"partitions":{"hex":["0103","ff00"],"length":10}}"#).unwrap();
        let mut second = vec![true; 8];
        second.extend([false, false]);
        assert_eq!(
            pool.partitions,
            vec![
                vec![
                    true, false, false, false, false, false, false, false, true, true
                ],
                second
            ]
        );
        let compact = serde_json::to_string(&pool).unwrap();
        assert_eq!(
            compact,
            r#"{"partitions":{"length":10,"hex":["0103","ff00"]}}"#
        );
    }
}
