//! Shared byte-serialization error, constants, and readers for all variants.

use thiserror::Error;

/// Error returned by every variant's `from_bytes` (aliased per variant).
#[derive(Error, Debug)]
pub enum DeserializeError {
    #[error(
        "Byte stream too short while reading {field}: need {needed} more byte(s), have {actual}"
    )]
    UnexpectedEof {
        field: &'static str,
        needed: usize,
        actual: usize,
    },

    #[error("Not a heavykeeper sketch: bad magic bytes {actual:02x?} (expected {expected:02x?})")]
    BadMagic { expected: [u8; 4], actual: [u8; 4] },

    #[error("Payload is a different sketch variant: got tag {actual} (expected {expected})")]
    WrongVariant { expected: u8, actual: u8 },

    #[error("Hasher mismatch: seed produces probe {actual} but payload holds {expected} (wrong seed passed to from_bytes)")]
    HasherMismatch { expected: u64, actual: u64 },

    #[error("Unsupported serialization version {version} (this build expects {expected})")]
    UnsupportedVersion { version: u8, expected: u8 },

    #[error("Invalid {field} value: {detail}")]
    InvalidField { field: &'static str, detail: String },

    #[error("Length mismatch for {field}: payload holds {actual} but expected {expected}")]
    LengthMismatch {
        field: &'static str,
        actual: usize,
        expected: usize,
    },

    #[error("{count} unexpected trailing byte(s) after the sketch payload")]
    TrailingBytes { count: usize },
}

/// Magic tag at the start of every serialized sketch (`b"HVYK"`).
pub(crate) const MAGIC: [u8; 4] = *b"HVYK";
/// On-disk format version. Bump whenever the byte layout changes.
pub(crate) const VERSION: u8 = 1;
/// Probe hashed at serialize time to detect a wrong seed on load.
pub(crate) const SERIALIZE_HASHER_PROBE: &[u8] = b"heavykeeper-serialize-hasher-probe";
/// Bytes per serialized cell: `(fingerprint: u64, count: u64)`.
pub(crate) const CELL_SIZE: usize = 16;
/// Bytes in a serialized `Xoshiro256PlusPlus` state (256-bit, little-endian).
pub(crate) const RNG_STATE_SIZE: usize = 32;

/// Narrow a `u64` to `usize`, erroring on overflow.
pub(crate) fn decoded_usize(value: u64, field: &'static str) -> Result<usize, DeserializeError> {
    usize::try_from(value).map_err(|_| DeserializeError::InvalidField {
        field,
        detail: format!("value {value} exceeds usize range on this platform"),
    })
}

/// Bounds-checked read of `n` bytes at `*pos`, advancing it.
pub(crate) fn take<'a>(
    bytes: &'a [u8],
    pos: &mut usize,
    n: usize,
    field: &'static str,
) -> Result<&'a [u8], DeserializeError> {
    let available = bytes.len().saturating_sub(*pos);
    if available < n {
        return Err(DeserializeError::UnexpectedEof {
            field,
            needed: n,
            actual: available,
        });
    }
    let slice = &bytes[*pos..*pos + n];
    *pos += n;
    Ok(slice)
}

/// Read a little-endian `u64` at `*pos`, advancing it.
pub(crate) fn take_u64(
    bytes: &[u8],
    pos: &mut usize,
    field: &'static str,
) -> Result<u64, DeserializeError> {
    Ok(u64::from_le_bytes(
        take(bytes, pos, 8, field)?
            .try_into()
            .expect("slice is 8 bytes"),
    ))
}
