//! The `table.info` marker (issue #23): every casacore table directory
//! carries one, so casacore-based consumers can recognise what a table is
//! (`Table::info()` is how a tool sees `Type = Measurement Set`).  casacure
//! wrote it for images only — a freshly created table had none, and
//! python-casacore reading casacure output reported `{'type': '', ...}`
//! where casacore's own output says `Measurement Set`.
//!
//! These tests pin the format against casacore's own bytes: the
//! casacore-written golden fixture `PARENT.tab` tracked in this repo, and
//! disko's casacore-written `test.ms` readme, byte for byte.

#[path = "../src/testdir.rs"]
#[allow(dead_code)]
mod testdir;

use casacure::record::{DataType, RecordValue, TableRecord};
use casacure::tableinfo::TableInfo;
use casacure::{ColumnDesc, ColumnKind, WritableTable};

/// The repo's casacore-written golden plain-table marker.
const GOLDEN_PLAIN: &[u8] = b"Type = \nSubType = \n\n";

/// The main-table marker of the casacore-written `test.ms` in disko.
const GOLDEN_MS: &[u8] = b"Type = Measurement Set\nSubType = \n\n\
This is a MeasurementSet Table holding measurements from a Telescope\n";

fn empty_record() -> TableRecord {
    TableRecord {
        desc: Default::default(),
        record_type: 0,
        values: Vec::new(),
    }
}

fn one_scalar_column() -> casacure::TableDesc {
    casacure::TableDesc {
        name: String::new(),
        version: String::new(),
        comment: String::new(),
        keywords: empty_record(),
        private_keywords: empty_record(),
        columns: vec![ColumnDesc {
            name: "VALUE".into(),
            comment: String::new(),
            data_type: DataType::Double,
            data_manager_type: "StandardStMan".into(),
            data_manager_group: "SSM".into(),
            options: 0,
            ndim: -1,
            shape: None,
            max_length: 0,
            keywords: empty_record(),
            kind: ColumnKind::Scalar(RecordValue::Double(0.0)),
            tile_shape: None,
        }],
    }
}

fn info_bytes(dir: &std::path::Path) -> Vec<u8> {
    std::fs::read(dir.join("table.info")).unwrap()
}

#[test]
fn create_writes_the_casacore_plain_marker() {
    let dir = testdir::TestDir::new("tableinfo-create".to_string());
    let values: Vec<Vec<RecordValue>> = vec![vec![RecordValue::Double(1.0)]];
    casacure::create_table(&dir, &one_scalar_column(), &values).unwrap();
    assert_eq!(info_bytes(&dir), GOLDEN_PLAIN);
    // And it parses back to the empty default.
    assert_eq!(TableInfo::read(&dir).unwrap(), TableInfo::default());
}

#[test]
fn created_bytes_match_the_casacore_golden_fixture() {
    // PARENT.tab is casacore-written ground truth tracked in this repo;
    // the created file must diff empty against it.
    let golden = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../PARENT.tab/table.info"
    ))
    .unwrap();
    assert_eq!(golden, GOLDEN_PLAIN);
}

#[test]
fn default_ms_marks_the_main_table_measurement_set() {
    let base = testdir::TestDir::new("tableinfo-ms".to_string());
    let path = base.join("test.ms");
    casacure::ms::default_ms(&path, None, None).unwrap();
    assert_eq!(info_bytes(&path), GOLDEN_MS);
    // The subtables stay plain, like casacore's MS trees.
    assert_eq!(info_bytes(&path.join("ANTENNA")), GOLDEN_PLAIN);
    let info = TableInfo::read(&path).unwrap();
    assert_eq!(info.r#type, "Measurement Set");
    assert_eq!(
        info.readme,
        "This is a MeasurementSet Table holding measurements from a Telescope\n"
    );
}

#[test]
fn a_whole_table_rewrite_preserves_the_existing_marker() {
    // The rewrite (create_table again over the directory) keeps the file
    // it already has: the marker belongs to the table's creator.
    let dir = testdir::TestDir::new("tableinfo-rewrite".to_string());
    let mut wt = WritableTable::create(&dir, one_scalar_column());
    wt.addrows(3);
    wt.flush().unwrap();
    TableInfo {
        r#type: "Custom".to_string(),
        sub_type: "Sub".to_string(),
        readme: "mine\n".to_string(),
    }
    .write(&dir)
    .unwrap();

    wt.removecol(0);
    wt.flush().unwrap();
    assert!(
        wt.full_rewrites() >= 1,
        "the flush after removecols takes the whole-table rewrite"
    );
    assert_eq!(info_bytes(&dir), b"Type = Custom\nSubType = Sub\n\nmine\n");
}

#[test]
fn put_then_read_round_trips_through_the_file() {
    let dir = testdir::TestDir::new("tableinfo-putread".to_string());
    let mut wt = WritableTable::create(&dir, one_scalar_column());
    wt.flush().unwrap();
    drop(wt);

    let mut info = TableInfo::read(&dir).unwrap();
    info.r#type = "Measurement Set".into();
    // A readme without its trailing newline is stored as whole lines.
    info.readme = "written by the round trip".into();
    info.write(&dir).unwrap();

    let read = TableInfo::read(&dir).unwrap();
    assert_eq!(read.r#type, "Measurement Set");
    assert_eq!(read.readme, "written by the round trip\n");
    assert_eq!(
        info_bytes(&dir),
        b"Type = Measurement Set\nSubType = \n\nwritten by the round trip\n"
    );
}
