pub mod codec;
pub mod predictor_model;

use std::{fmt::Write, io::Write as IoWriter};

use flate2::{
    write::{GzDecoder, GzEncoder},
    Compression,
};
use thiserror::Error;

use crate::{
    io::{self, BitInputStore, BitOutputStore},
    INT4_NULL_CODE,
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("compression error {0}")]
    Compress(String),
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    #[error("fmt error {0}")]
    Fmt(#[from] std::fmt::Error),
    #[error("io error {0}")]
    GvrsIo(#[from] io::Error),
}

const MASK: i8 = 0xff_u8 as i8;

pub trait CompressionEncoder {
    fn encode(
        &mut self,
        codec_index: i32,
        n_rows: i32,
        n_cols: i32,
        values: &[i32],
    ) -> Result<Vec<i8>>;

    fn encode_floats(
        &mut self,
        codec_index: i32,
        n_rows: i32,
        n_cols: i32,
        values: &[f32],
    ) -> Result<Vec<i8>>;

    fn implements_floating_point_encoding(&self) -> bool;

    fn implements_integer_encoding(&self) -> bool;
}

pub trait CompressionDecoder {
    fn decode(&mut self, n_rows: i32, n_columns: i32, packing: &[i8]) -> Result<Vec<i32>>;

    fn analyze(&mut self, n_rows: i32, n_columns: i32, packing: &[i8]) -> Result<()>;

    fn report_analysis_data(&self, writer: impl Write, n_tiles_in_raster: i32) -> Result<()>;

    fn clear_analysis_data(&mut self) -> Result<()>;

    fn decode_floats(&mut self, n_rows: i32, n_columns: i32, packing: &[i8]) -> Result<Vec<f32>>;
}
