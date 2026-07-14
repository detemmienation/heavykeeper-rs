//! Shared byte-serialization error, constants, and readers for all variants.

use ahash::RandomState;
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

/// A forward-only cursor over a serialized payload. Every read is
/// bounds-checked and advances the cursor, so `from_bytes` never touches raw
/// offsets and a truncated stream fails with a precise `UnexpectedEof`.
pub(crate) struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Read `n` bytes, advancing the cursor.
    pub(crate) fn take(&mut self, n: usize, field: &'static str) -> Result<&'a [u8], DeserializeError> {
        let available = self.bytes.len().saturating_sub(self.pos);
        if available < n {
            return Err(DeserializeError::UnexpectedEof {
                field,
                needed: n,
                actual: available,
            });
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    /// Read a fixed-size byte array.
    pub(crate) fn take_array<const N: usize>(
        &mut self,
        field: &'static str,
    ) -> Result<[u8; N], DeserializeError> {
        Ok(self.take(N, field)?.try_into().expect("slice is N bytes"))
    }

    /// Read a single byte.
    pub(crate) fn take_u8(&mut self, field: &'static str) -> Result<u8, DeserializeError> {
        Ok(self.take(1, field)?[0])
    }

    /// Read a little-endian `u64`.
    pub(crate) fn take_u64(&mut self, field: &'static str) -> Result<u64, DeserializeError> {
        Ok(u64::from_le_bytes(self.take_array::<8>(field)?))
    }

    /// Read a little-endian `u64` and narrow it to `usize`, erroring on overflow.
    pub(crate) fn take_usize(&mut self, field: &'static str) -> Result<usize, DeserializeError> {
        let value = self.take_u64(field)?;
        usize::try_from(value).map_err(|_| DeserializeError::InvalidField {
            field,
            detail: format!("value {value} exceeds usize range on this platform"),
        })
    }

    /// Verify the fixed header shared by every variant: magic, `variant` tag,
    /// version, and the hasher probe. Rebuilds the hasher from `seed` and
    /// rejects a wrong seed before any geometry is parsed.
    pub(crate) fn read_header(&mut self, variant: u8, seed: u64) -> Result<(), DeserializeError> {
        let magic = self.take_array::<4>("magic")?;
        if magic != MAGIC {
            return Err(DeserializeError::BadMagic {
                expected: MAGIC,
                actual: magic,
            });
        }
        let got_variant = self.take_u8("variant")?;
        if got_variant != variant {
            return Err(DeserializeError::WrongVariant {
                expected: variant,
                actual: got_variant,
            });
        }
        let version = self.take_u8("version")?;
        if version != VERSION {
            return Err(DeserializeError::UnsupportedVersion {
                version,
                expected: VERSION,
            });
        }
        let expected_probe = self.take_u64("hasher_probe")?;
        let actual_probe = RandomState::with_seeds(seed, seed, seed, seed).hash_one(SERIALIZE_HASHER_PROBE);
        if actual_probe != expected_probe {
            return Err(DeserializeError::HasherMismatch {
                expected: expected_probe,
                actual: actual_probe,
            });
        }
        Ok(())
    }

    /// Reject any bytes left after the payload.
    pub(crate) fn finish(&self) -> Result<(), DeserializeError> {
        if self.pos != self.bytes.len() {
            return Err(DeserializeError::TrailingBytes {
                count: self.bytes.len() - self.pos,
            });
        }
        Ok(())
    }
}
