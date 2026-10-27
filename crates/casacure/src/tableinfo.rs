//! The `table.info` file (casacore's `TableInfo`): the type / subtype /
//! readme marker every casacore table directory carries (issue #23).
//!
//! casacore creates the file whenever it creates a table — a plain table
//! gets `Type = \nSubType = \n\n`, a Measurement Set's main table gets
//! `Type = Measurement Set` plus the MS readme — and consumers branch on
//! it (`Table::info()` is how a tool recognises an MS). The image path
//! wrote it from the start (a PagedImage is `Type = Image`); tables now
//! do the same, through this one module.
//!
//! The serialisation is byte-exact casacore (probed against
//! python-casacore 3.8.1 and the casacore-written golden fixtures
//! `PARENT.tab`, `SUB.tab` and disko's `test.ms`): two `key = value`
//! lines, then an unconditional blank line, then the readme kept as whole
//! lines (casacore appends a missing trailing newline at set time).

/// The readme casacore's Measurement Set creation writes on the main table.
pub const MS_README: &str =
    "This is a MeasurementSet Table holding measurements from a Telescope\n";

/// The parsed contents of a table directory's `table.info`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TableInfo {
    /// The `Type =` value (empty for a plain table, `"Measurement Set"`
    /// for an MS main table, `"Image"` for a PagedImage).
    pub r#type: String,
    /// The `SubType =` value.
    pub sub_type: String,
    /// The readme text: everything after the separating blank line, kept
    /// verbatim including its trailing newline (empty when there is no
    /// readme).
    pub readme: String,
}

impl TableInfo {
    /// The `table.info` of a Measurement Set's main table — what
    /// casacore's MS creation writes (subtables stay plain).
    pub fn ms_main() -> TableInfo {
        TableInfo {
            r#type: "Measurement Set".to_string(),
            sub_type: String::new(),
            readme: MS_README.to_string(),
        }
    }

    /// Read `<dir>/table.info`. A missing file reads as the empty default
    /// (casacore, probed: a table whose `table.info` was deleted answers
    /// `info()` with empty strings and does not recreate the file).
    pub fn read(dir: &std::path::Path) -> std::io::Result<TableInfo> {
        match std::fs::read(dir.join("table.info")) {
            Ok(bytes) => Ok(parse(&String::from_utf8_lossy(&bytes))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(TableInfo::default()),
            Err(e) => Err(e),
        }
    }

    /// Write `<dir>/table.info` with casacore's exact bytes (atomic, so a
    /// concurrent `info()` reader sees the whole old or whole new file).
    pub fn write(&self, dir: &std::path::Path) -> std::io::Result<()> {
        // casacore stores the readme as whole lines (`TableInfo::
        // setReadme` appends the missing newline); serialise to the same.
        let mut readme = self.readme.clone();
        if !readme.is_empty() && !readme.ends_with('\n') {
            readme.push('\n');
        }
        let text = format!(
            "Type = {}\nSubType = {}\n\n{}",
            self.r#type, self.sub_type, readme
        );
        std::fs::create_dir_all(dir)?;
        crate::datafile::write_atomic(&dir.join("table.info"), text.as_bytes())
    }
}

/// Parse a `table.info` file's text. `Type`/`SubType` are the values of
/// the first two lines; the readme is everything after the first blank
/// line (casacore always writes the separator), kept verbatim.
fn parse(text: &str) -> TableInfo {
    let mut info = TableInfo::default();
    let mut lines = text.split_inclusive('\n');
    if let Some(line) = lines.next() {
        info.r#type = value_of(line, "Type");
    }
    if let Some(line) = lines.next() {
        info.sub_type = value_of(line, "SubType");
    }
    let rest: String = lines.collect();
    info.readme = rest.strip_prefix('\n').unwrap_or(&rest).to_string();
    info
}

/// The value of one `key = value` line: the text after `key =` and one
/// separating space, kept verbatim (an absent or empty value is `""`).
fn value_of(line: &str, key: &str) -> String {
    let prefix = format!("{key} = ");
    let bare = format!("{key} =");
    let line = line.trim_end_matches('\n');
    if let Some(v) = line.strip_prefix(&prefix) {
        return v.to_string();
    }
    line.strip_prefix(&bare).unwrap_or_default().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_table_bytes_match_casacore() {
        // The casacore-written golden fixture PARENT.tab, byte for byte.
        let golden = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../PARENT.tab/table.info"
        ))
        .unwrap();
        assert_eq!(golden, b"Type = \nSubType = \n\n");
        let dir = crate::testdir::TestDir::new("tableinfo-plain".to_string());
        TableInfo::default().write(&dir).unwrap();
        assert_eq!(std::fs::read(dir.join("table.info")).unwrap(), golden);
    }

    #[test]
    fn ms_bytes_match_the_casacore_written_fixture() {
        let dir = crate::testdir::TestDir::new("tableinfo-ms".to_string());
        TableInfo::ms_main().write(&dir).unwrap();
        assert_eq!(
            std::fs::read(dir.join("table.info")).unwrap(),
            b"Type = Measurement Set\nSubType = \n\n\
              This is a MeasurementSet Table holding measurements from a Telescope\n"
        );
    }

    #[test]
    fn image_marker_bytes_are_unchanged() {
        // The bytes images/write.rs has always written (tests assert the
        // file's presence through python-casacore reads).
        let dir = crate::testdir::TestDir::new("tableinfo-image".to_string());
        TableInfo {
            r#type: "Image".to_string(),
            ..Default::default()
        }
        .write(&dir)
        .unwrap();
        assert_eq!(
            std::fs::read(dir.join("table.info")).unwrap(),
            b"Type = Image\nSubType = \n\n"
        );
    }

    #[test]
    fn write_normalises_a_readme_missing_its_newline() {
        // casacore stores the readme as whole lines: `putinfo` with
        // readme 'hello' leaves `'hello\n'` in the file and in `info()`.
        let dir = crate::testdir::TestDir::new("tableinfo-newline".to_string());
        TableInfo {
            r#type: "T".to_string(),
            sub_type: "S".to_string(),
            readme: "hello".to_string(),
        }
        .write(&dir)
        .unwrap();
        assert_eq!(
            std::fs::read(dir.join("table.info")).unwrap(),
            b"Type = T\nSubType = S\n\nhello\n"
        );
        assert_eq!(
            TableInfo::read(&dir).unwrap(),
            TableInfo {
                r#type: "T".to_string(),
                sub_type: "S".to_string(),
                readme: "hello\n".to_string(),
            }
        );
    }

    #[test]
    fn round_trips_the_readme_verbatim() {
        let dir = crate::testdir::TestDir::new("tableinfo-roundtrip".to_string());
        let info = TableInfo {
            r#type: "Measurement Set".to_string(),
            sub_type: "MS v2".to_string(),
            readme: "line one\nline two\n".to_string(),
        };
        info.write(&dir).unwrap();
        assert_eq!(TableInfo::read(&dir).unwrap(), info);
    }

    #[test]
    fn read_tolerates_a_missing_file() {
        let dir = crate::testdir::TestDir::new("tableinfo-missing".to_string());
        assert_eq!(TableInfo::read(&dir).unwrap(), TableInfo::default());
    }

    #[test]
    fn parses_the_fixture_shapes() {
        assert_eq!(
            parse("Type = \nSubType = \n\n"),
            TableInfo::default(),
            "plain table"
        );
        assert_eq!(
            parse("Type = Measurement Set\nSubType = \n\nThis is a MeasurementSet Table holding measurements from a Telescope\n"),
            TableInfo::ms_main(),
            "MS main table"
        );
        // A truncated file with no separating blank line: the rest is the
        // readme (lenient, like a reader of casacore's own files).
        assert_eq!(
            parse("Type = X\nSubType = Y\n"),
            TableInfo {
                r#type: "X".to_string(),
                sub_type: "Y".to_string(),
                readme: String::new(),
            }
        );
    }
}
