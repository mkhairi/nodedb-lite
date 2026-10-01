// SPDX-License-Identifier: Apache-2.0

//! Serialized row counting rejects oversized inputs without allocating their encoding.

use crate::error::LiteError;
use nodedb_types::Value;
use std::collections::HashMap;
use std::io::{self, Write};

pub(crate) const PAGE_RECORDS: usize = 128;
pub(crate) const PAGE_BYTES: usize = 8 * 1024 * 1024;

struct Counter {
    remaining: usize,
    exceeded: bool,
}

impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.remaining {
            self.exceeded = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "serialized text row exceeds byte budget",
            ));
        }
        self.remaining -= bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn check_row_budget(
    collection: &str,
    id: &str,
    fields: &HashMap<String, Value>,
    max_bytes: usize,
) -> Result<(), LiteError> {
    let mut counter = Counter {
        remaining: max_bytes.saturating_sub(id.len()),
        exceeded: id.len() > max_bytes,
    };
    let encoded = zerompk::write_msgpack(&mut counter, fields);
    if counter.exceeded {
        return Err(LiteError::Backpressure {
            detail: format!(
                "text row '{collection}'/'{id}' exceeds {max_bytes} serialized bytes: reduce the document"
            ),
        });
    }
    encoded.map_err(|error| LiteError::Serialization {
        detail: format!("text row '{collection}'/'{id}': {error}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_byte_budget_rejects_oversized_strings_without_an_encoding_buffer() {
        let fields = HashMap::from([("body".into(), Value::String("abcdef".into()))]);
        assert!(check_row_budget("docs", "id", &fields, 100).is_ok());
        assert!(matches!(
            check_row_budget("docs", "id", &fields, 5),
            Err(LiteError::Backpressure { .. })
        ));
    }
}
