use std::{fmt::Write, io::Write as IoWriter};

use flate2::{
    write::{GzDecoder, GzEncoder},
    Compression,
};
use thiserror::Error;

use crate::{io::BitOutputStore, INT4_NULL_CODE};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("compression error {0}")]
    Compress(String),
    #[error("io error {0}")]
    Io(#[from] std::io::Error),
    #[error("fmt error {0}")]
    Fmt(#[from] std::fmt::Error),
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

    fn decode_floats(&self, n_rows: i32, n_columns: i32, packing: &[i8]) -> Result<Vec<f32>>;
}

pub trait PredictorModel {
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        seed: i32,
        n_rows: i32,
        n_columns: i32,
        encoding: &[i8],
        offset: i32,
        length: i32,
        output: &mut [i32],
    ) -> Result<()>;

    fn encode(
        &mut self,
        n_rows: i32,
        n_columns: i32,
        values: &[i32],
        encoding: &mut [i8],
    ) -> Result<i32>;

    fn is_null_data_supported(&self) -> bool;

    fn seed(&self) -> i32;

    fn predictor_type(&self) -> PredictorModelType;
}

#[derive(Clone, Copy)]
pub enum PredictorModelType {
    None,
    Differencing,
    Linear,
    Triangle,
    DifferencingWithNulls,
}

impl PredictorModelType {
    pub fn label(&self) -> &'static str {
        match self {
            PredictorModelType::None => "None",
            PredictorModelType::Differencing => "Differencing",
            PredictorModelType::Linear => "Linear",
            PredictorModelType::Triangle => "Triangle",
            PredictorModelType::DifferencingWithNulls => "DifferencingWithNulls",
        }
    }

    pub fn code_value(&self) -> i8 {
        match self {
            PredictorModelType::None => 0_i8,
            PredictorModelType::Differencing => 1_i8,
            PredictorModelType::Linear => 2_i8,
            PredictorModelType::Triangle => 3_i8,
            PredictorModelType::DifferencingWithNulls => 4_i8,
        }
    }
}

impl Default for PredictorModelType {
    fn default() -> Self {
        PredictorModelType::None
    }
}

impl From<i8> for PredictorModelType {
    fn from(code_value: i8) -> Self {
        match code_value {
            1 => PredictorModelType::Differencing,
            2 => PredictorModelType::Linear,
            3 => PredictorModelType::Triangle,
            4 => PredictorModelType::DifferencingWithNulls,
            _ => PredictorModelType::None,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct CodecStats {
    pub pc_type: PredictorModelType,
    pub n_tiles_counted: i64,
    pub n_bytes_total: i64,
    pub n_symbols_total: i64,
    pub n_bits_overhead_total: i64,
    pub n_m32_counted: i64,
    pub sum_length_m32: i64,
    pub sum_observed_m32: i64,
    pub sum_entropy_m32: f64,
}

impl CodecStats {
    pub fn new(pc_type: PredictorModelType) -> Self {
        CodecStats {
            pc_type,
            ..Default::default()
        }
    }

    pub fn label(&self) -> &'static str {
        self.pc_type.label()
    }

    pub fn add_to_counts(
        &mut self,
        n_bytes_for_tile: i32,
        n_symbols_in_tile: i32,
        n_bits_overhead: i32,
    ) {
        self.n_tiles_counted += 1;
        self.n_bytes_total += n_bytes_for_tile as i64;
        self.n_symbols_total += n_symbols_in_tile as i64;
        self.n_bits_overhead_total += n_bits_overhead as i64;
    }

    pub fn add_counts_for_m32(&mut self, n_m32: i32, m32: &[u8]) -> Result<()> {
        if n_m32 == 0 {
            return Ok(());
        }

        self.n_m32_counted += 1;
        self.sum_length_m32 += n_m32 as i64;

        let mut observed: Vec<i32> = Vec::with_capacity(256);
        let mut m_count: Vec<i32> = Vec::with_capacity(256);

        for i in 0..n_m32 {
            let index: i32 = m32[i as usize] as i32;
            m_count[index as usize] += 1;
            observed[index as usize] = 1;
        }

        for i in observed.iter().take(256) {
            self.sum_observed_m32 += *i as i64;
        }

        let d = n_m32 as f64;
        let mut s = 0_f64;
        for i in m_count.iter().take(256) {
            if *i > 0 {
                let p = *i as f64 / d;
                s += p * p.log2() / 2.0_f64.log2();
            }
        }

        self.sum_entropy_m32 = s;

        Ok(())
    }

    pub fn entropy(&self) -> f64 {
        if self.n_m32_counted == 0 {
            return 0_f64;
        }

        self.sum_entropy_m32 / self.n_m32_counted as f64
    }

    pub fn clear(&mut self) {
        self.n_tiles_counted = 0;
        self.n_bytes_total = 0;
        self.n_symbols_total = 0;
        self.n_bits_overhead_total = 0;
    }

    pub fn bits_per_symbol(&self) -> f64 {
        if self.n_m32_counted == 0 {
            return 0_f64;
        }

        self.sum_length_m32 as f64 / self.n_m32_counted as f64
    }

    pub fn average_mcode_length(&self) -> f64 {
        if self.n_m32_counted == 0 {
            return 0_f64;
        }

        self.sum_length_m32 as f64 / self.n_m32_counted as f64
    }

    pub fn average_observed_mcodes(&self) -> f64 {
        if self.n_tiles_counted > 0 {
            return self.sum_observed_m32 as f64 / self.n_tiles_counted as f64;
        }

        0_f64
    }

    pub fn average_overhead(&self) -> f64 {
        if self.n_tiles_counted == 0 {
            return 0_f64;
        }

        self.n_bits_overhead_total as f64 / self.n_tiles_counted as f64
    }

    pub fn average_length(&self) -> f64 {
        if self.n_tiles_counted == 0 {
            return 0_f64;
        }

        self.n_bytes_total as f64 / self.n_tiles_counted as f64
    }
}

#[derive(Default)]
pub struct CodecM32 {
    buffer: Vec<i8>,
    buffer_limit: i32,
    offset: i32,
    offset0: i32,
}

impl CodecM32 {
    pub const MAX_BYTES_PER_VALUE: i32 = 6;
    const LO_MASK: i32 = 0b0111_1111;
    const HI_BIT: i32 = 0b1000_0000;
    const SEGMENT_BASE_VALUE: [i32; 5] = [127, 255, 16639, 2113791, 270549247];

    pub fn new(buffer: &[i8], offset: i32, length: i32) -> Self {
        CodecM32 {
            buffer: buffer.into(),
            buffer_limit: offset + length,
            offset,
            offset0: offset,
        }
    }

    pub fn from_symbol_count(symbol_count: i32) -> Self {
        let buffer_length = symbol_count * CodecM32::MAX_BYTES_PER_VALUE;

        CodecM32 {
            buffer: vec![0; buffer_length as usize],
            buffer_limit: buffer_length,
            ..Default::default()
        }
    }

    pub fn rewind(&mut self) {
        self.offset = self.offset0;
    }

    pub fn encoding(&self) -> Vec<i8> {
        let n = self.offset - self.offset0;
        let mut b = vec![0; n as usize];

        b.copy_from_slice(&self.buffer[self.offset0 as usize..]);

        b
    }

    pub fn encoded_length(&self) -> i32 {
        self.offset - self.offset0
    }

    pub fn encode(&mut self, value: i32) {
        #[allow(unused_assignments)]
        let mut abs_value = 0;

        if value < 0 {
            if value == i32::MIN {
                self.buffer[self.offset as usize] = -128_i8;
                self.offset += 1;
                return;
            } else if value > -127 {
                self.buffer[self.offset as usize] = value as i8;
                self.offset += 1;
                return;
            }
            self.buffer[self.offset as usize] = -127_i8;
            self.offset += 1;
            abs_value = -value;
        } else {
            if value < 127 {
                self.buffer[self.offset as usize] = value as i8;
                self.offset += 1;
                return;
            }
            self.buffer[self.offset as usize] = 127_i8;
            self.offset += 1;
            abs_value = value;
        }

        if abs_value <= 254 {
            let delta = abs_value - 127;
            self.buffer[self.offset as usize] = delta as i8;
            self.offset += 1;
        } else if abs_value <= 16638 {
            let delta = abs_value - 255;
            self.buffer[self.offset as usize] =
                (((delta >> 7) & CodecM32::LO_MASK) | CodecM32::HI_BIT) as i8;
            self.offset += 1;
            self.buffer[self.offset as usize] = (delta & CodecM32::LO_MASK) as i8;
            self.offset += 1;
        } else if abs_value <= 2113790 {
            let delta = abs_value - 16639;
            self.buffer[self.offset as usize] =
                (((delta >> 14) & CodecM32::LO_MASK) | CodecM32::HI_BIT) as i8;
            self.offset += 1;
            self.buffer[self.offset as usize] =
                (((delta >> 7) & CodecM32::LO_MASK) | CodecM32::HI_BIT) as i8;
            self.offset += 1;
            self.buffer[self.offset as usize] = (delta & CodecM32::LO_MASK) as i8;
            self.offset += 1;
        } else if abs_value <= 270549246 {
            let delta = abs_value - 2113791;
            self.buffer[self.offset as usize] =
                (((delta >> 21) & CodecM32::LO_MASK) | CodecM32::HI_BIT) as i8;
            self.offset += 1;
            self.buffer[self.offset as usize] =
                (((delta >> 14) & CodecM32::LO_MASK) | CodecM32::HI_BIT) as i8;
            self.offset += 1;
            self.buffer[self.offset as usize] =
                (((delta >> 7) & CodecM32::LO_MASK) | CodecM32::HI_BIT) as i8;
            self.offset += 1;
            self.buffer[self.offset as usize] = (delta & CodecM32::LO_MASK) as i8;
            self.offset += 1;
        } else {
            let delta = abs_value - 270549247;
            self.buffer[self.offset as usize] =
                (((delta >> 28) & CodecM32::LO_MASK) | CodecM32::HI_BIT) as i8;
            self.offset += 1;
            self.buffer[self.offset as usize] =
                (((delta >> 21) & CodecM32::LO_MASK) | CodecM32::HI_BIT) as i8;
            self.offset += 1;
            self.buffer[self.offset as usize] =
                (((delta >> 14) & CodecM32::LO_MASK) | CodecM32::HI_BIT) as i8;
            self.offset += 1;
            self.buffer[self.offset as usize] =
                (((delta >> 7) & CodecM32::LO_MASK) | CodecM32::HI_BIT) as i8;
            self.offset += 1;
            self.buffer[self.offset as usize] = (delta & CodecM32::LO_MASK) as i8;
            self.offset += 1;
        }
    }

    pub fn decode(&mut self) -> i32 {
        let symbol = self.buffer[self.offset as usize];
        self.offset += 1;
        if symbol == -128 {
            return i32::MIN;
        } else if -127 < symbol && symbol < 127 {
            return symbol as i32;
        }

        let mut delta = 0_i32;
        for i in CodecM32::SEGMENT_BASE_VALUE.iter() {
            let sample = self.buffer[self.offset as usize];
            self.offset += 1;
            delta = (delta << 7) | (sample as i32 & CodecM32::LO_MASK);
            if sample as i32 & CodecM32::HI_BIT == 0 {
                if symbol == -127 {
                    delta = -delta - i;
                } else {
                    delta += i;
                }
                break;
            }
        }

        delta
    }

    pub fn remaining(&self) -> i32 {
        self.buffer_limit - self.offset
    }
}

pub struct CodecDeflate {
    pub predictor: Vec<Box<dyn PredictorModel>>,
    pub codec_stats: Vec<CodecStats>,
}

impl Default for CodecDeflate {
    fn default() -> Self {
        CodecDeflate {
            predictor: vec![
                Box::new(PredictorModelDifferencing::default()),
                Box::new(PredictorModelLinear::default()),
                Box::new(PredictorModelTriangle::default()),
                Box::new(PredictorModelDifferencingWithNulls::default()),
            ],
            codec_stats: Vec::default(),
        }
    }
}

impl CompressionDecoder for CodecDeflate {
    fn decode(&mut self, n_rows: i32, n_columns: i32, packing: &[i8]) -> Result<Vec<i32>> {
        let seed = (packing[2] & MASK) as i32
            | ((packing[3] & MASK) as i32) << 8
            | ((packing[4] & MASK) as i32) << 16
            | ((packing[5] & MASK) as i32) << 24;

        let n_m32 = (packing[6] & MASK) as i32
            | ((packing[7] & MASK) as i32) << 8
            | ((packing[8] & MASK) as i32) << 16
            | ((packing[9] & MASK) as i32) << 24;

        let mut code_m32s = vec![0; n_m32 as usize];
        let mut decoder = GzDecoder::new(code_m32s);
        let test = decoder.write(
            &packing[10..packing.len() - 10]
                .iter()
                .map(|i| *i as u8)
                .collect::<Vec<_>>(),
        )?;
        code_m32s = decoder.finish()?;
        if test > 0 {
            let mut output = vec![0; (n_rows * n_columns) as usize];

            match PredictorModelType::from(packing[1] as i8) {
                PredictorModelType::Differencing => {
                    let pcc = PredictorModelDifferencing::default();
                    pcc.decode(
                        seed,
                        n_rows,
                        n_columns,
                        &code_m32s.iter().map(|i| *i as i8).collect::<Vec<_>>(),
                        0,
                        n_m32,
                        &mut output,
                    )?;
                }
                PredictorModelType::Linear => {
                    let pcc = PredictorModelLinear::default();
                    pcc.decode(
                        seed,
                        n_rows,
                        n_columns,
                        &code_m32s.iter().map(|i| *i as i8).collect::<Vec<_>>(),
                        0,
                        n_m32,
                        &mut output,
                    )?;
                }
                PredictorModelType::Triangle => {
                    let pcc = PredictorModelTriangle::default();
                    pcc.decode(
                        seed,
                        n_rows,
                        n_columns,
                        &code_m32s.iter().map(|i| *i as i8).collect::<Vec<_>>(),
                        0,
                        n_m32,
                        &mut output,
                    )?;
                }
                PredictorModelType::DifferencingWithNulls => {
                    let pcc = PredictorModelDifferencingWithNulls::default();
                    pcc.decode(
                        seed,
                        n_rows,
                        n_columns,
                        &code_m32s.iter().map(|i| *i as i8).collect::<Vec<_>>(),
                        0,
                        n_m32,
                        &mut output,
                    )?;
                }
                _ => {
                    return Err(Error::Compress(
                        "unknown PredictorCorrector type".to_string(),
                    ))
                }
            }

            return Ok(output);
        }

        Ok(Vec::default())
    }

    fn analyze(&mut self, n_rows: i32, n_columns: i32, packing: &[i8]) -> Result<()> {
        if self.codec_stats.is_empty() {
            self.codec_stats
                .push(CodecStats::new(PredictorModelType::None));
        }

        let mut stats = self.codec_stats[(packing[1] & MASK) as usize];
        let n_values = n_rows * n_columns;
        stats.add_to_counts((packing.len() - 10) as i32, n_values, 0);

        let n_m32 = (packing[6] & MASK) as i32
            | ((packing[7] & MASK) as i32) << 8
            | ((packing[8] & MASK) as i32) << 16
            | ((packing[9] & MASK) as i32) << 24;

        let mut code_m32s = vec![0; n_m32 as usize];
        let mut decoder = GzDecoder::new(code_m32s);
        let test = decoder.write(
            &packing[10..packing.len() - 10]
                .iter()
                .map(|i| *i as u8)
                .collect::<Vec<_>>(),
        )?;
        code_m32s = decoder.finish()?;
        if test > 0 {
            stats.add_counts_for_m32(n_m32, &code_m32s)?;
        }

        Ok(())
    }

    fn report_analysis_data(&self, mut writer: impl Write, n_tiles_in_raster: i32) -> Result<()> {
        writer.write_str("Gridfour_Deflate                               Compressed Output    |       Predictor Residuals\n")?;
        if self.codec_stats.is_empty() || n_tiles_in_raster == 0 {
            writer.write_str("   Tiles Compressed:  0\n")?;
            return Ok(());
        }

        writer.write_str("  Predictor                Times Used        bits/sym    bits/tile  |  m32 avg-len   avg-unique  entropy\n")?;

        for stats in self.codec_stats.iter() {
            let label = stats.label();
            if label == "None" {
                continue;
            }
            let tile_count = stats.n_tiles_counted;
            let bits_per_symbol = stats.bits_per_symbol();
            let avg_bits_in_text = stats.average_length() * 8_f64;
            let avg_unique_symbols = stats.average_observed_mcodes();
            let avg_mcode_length = stats.average_mcode_length();
            let percent_tiles = 100.0 * (tile_count / n_tiles_in_raster as i64) as f64;
            let entropy = stats.entropy();
            writer.write_fmt(format_args!(
                "   {}-20.20s {} ({:.1} %)     {:.2}  {:.1}   | {:.1}      {:.1}    {:.2}\n",
                label,
                tile_count,
                percent_tiles,
                bits_per_symbol,
                avg_bits_in_text,
                avg_mcode_length,
                avg_unique_symbols,
                entropy,
            ))?;
        }

        Ok(())
    }

    fn clear_analysis_data(&mut self) -> Result<()> {
        self.codec_stats = Vec::default();
        Ok(())
    }

    fn decode_floats(&self, _n_rows: i32, _n_columns: i32, _packing: &[i8]) -> Result<Vec<f32>> {
        Ok(Vec::default())
    }
}

impl CompressionEncoder for CodecDeflate {
    fn encode(
        &mut self,
        codec_index: i32,
        n_rows: i32,
        n_cols: i32,
        values: &[i32],
    ) -> Result<Vec<i8>> {
        let mut contains_null_value = false;
        let mut contains_valid_data = false;

        for i in values.iter() {
            if *i == INT4_NULL_CODE {
                contains_null_value = true;
            } else {
                contains_valid_data = true;
            }
        }

        if !contains_valid_data {
            return Ok(Vec::default());
        }

        let mut m_code = vec![0; (CodecM32::MAX_BYTES_PER_VALUE * n_rows * n_cols) as usize];

        let mut result_length = i32::MAX;
        let mut result_bytes: Vec<i8> = Vec::default();

        for i in 0..self.predictor.len() {
            if contains_null_value {
                if !self.predictor[i].is_null_data_supported() {
                    continue;
                }
            } else if self.predictor[i].is_null_data_supported() {
                continue;
            }

            let m_code_length = self.predictor[i].encode(n_rows, n_cols, values, &mut m_code)?;
            if m_code_length > 0 {
                let test_bytes = self.compress(
                    codec_index,
                    self.predictor[i].as_ref(),
                    &m_code,
                    m_code_length,
                )?;
                if !test_bytes.is_empty() && test_bytes.len() < result_length as usize {
                    result_length = test_bytes.len() as i32;
                    result_bytes = test_bytes;
                }
            }
        }

        Ok(result_bytes)
    }

    fn encode_floats(
        &mut self,
        _codec_index: i32,
        _n_rows: i32,
        _n_cols: i32,
        _values: &[f32],
    ) -> Result<Vec<i8>> {
        Ok(Vec::default())
    }

    fn implements_floating_point_encoding(&self) -> bool {
        false
    }

    fn implements_integer_encoding(&self) -> bool {
        true
    }
}

impl CodecDeflate {
    fn compress(
        &self,
        codec_index: i32,
        pcc: &dyn PredictorModel,
        m_codes: &[i8],
        n_m32: i32,
    ) -> Result<Vec<i8>> {
        let seed = pcc.seed();
        let mut deflator_result = vec![0; (n_m32 + 128) as usize];
        let deflator_result_len = deflator_result.len();
        let mut encoder = GzEncoder::new(deflator_result, Compression::new(9));
        let d_n = encoder.write(
            &m_codes[10..deflator_result_len - 10]
                .iter()
                .map(|i| *i as u8)
                .collect::<Vec<_>>(),
        )?;

        encoder.flush()?;
        deflator_result = encoder.finish()?;

        if d_n == 0 {
            return Ok(Vec::default());
        }

        let mut deflator_result = deflator_result.iter().map(|i| *i as i8).collect::<Vec<_>>();
        deflator_result[0] = codec_index as i8;
        deflator_result[1] = pcc.predictor_type().code_value();
        deflator_result[2] = (seed & 0xff) as i8;
        deflator_result[3] = ((seed >> 8) & 0xff) as i8;
        deflator_result[4] = ((seed >> 16) & 0xff) as i8;
        deflator_result[5] = ((seed >> 24) & 0xff) as i8;
        deflator_result[6] = (n_m32 & 0xff) as i8;
        deflator_result[7] = ((n_m32 >> 8) & 0xff) as i8;
        deflator_result[8] = ((n_m32 >> 16) & 0xff) as i8;
        deflator_result[9] = ((n_m32 >> 24) & 0xff) as i8;
        let mut b = vec![0; d_n + 10];
        b.copy_from_slice(&deflator_result[..]);
        Ok(b)
    }
}

pub struct CodecFloat {
    n_cells_in_tile: i32,
    was_data_encoded: bool,
    s_total: SimpleStats,
    s_sign_bit: SimpleStats,
    s_exp: SimpleStats,
    s_m1_delta: SimpleStats,
    s_m2_delta: SimpleStats,
    s_m3_delta: SimpleStats,
}

impl CompressionEncoder for CodecFloat {
    fn encode(
        &mut self,
        codec_index: i32,
        n_rows: i32,
        n_cols: i32,
        values: &[i32],
    ) -> Result<Vec<i8>> {
        Err(Error::Compress(
            "attempt to encode an integral format not supported by this CODEC".to_string(),
        ))
    }

    fn encode_floats(
        &mut self,
        codec_index: i32,
        n_rows: i32,
        n_cols: i32,
        values: &[f32],
    ) -> Result<Vec<i8>> {
        self.n_cells_in_tile = n_rows * n_cols;
        self.was_data_encoded = true;

        let mut c = vec![0; values.len()];
        for (i, value) in values.iter().enumerate() {
            c[1] = value.to_bits();
        }
        let mut b_sign = BitOutputStore::default();

        todo!()
    }

    fn implements_floating_point_encoding(&self) -> bool {
        todo!()
    }

    fn implements_integer_encoding(&self) -> bool {
        todo!()
    }
}

impl CompressionDecoder for CodecFloat {
    fn decode(&mut self, n_rows: i32, n_columns: i32, packing: &[i8]) -> Result<Vec<i32>> {
        Err(Error::Compress(
            "attempt to decode an integral format not supported by this CODEC".to_string(),
        ))
    }

    fn analyze(&mut self, n_rows: i32, n_columns: i32, packing: &[i8]) -> Result<()> {
        self.n_cells_in_tile = n_rows * n_columns;
        self.was_data_encoded = true;
        let mut offset = 2;
        let mut n = Self::unpack_integer(packing, offset);
        self.s_sign_bit.add_count(n);
        offset += 4 + n;

        n = Self::unpack_integer(packing, offset);
        self.s_exp.add_count(n);
        offset += 4 + n;

        n = Self::unpack_integer(packing, offset);
        self.s_m1_delta.add_count(n);
        offset += 4 + n;

        n = Self::unpack_integer(packing, offset);
        self.s_m2_delta.add_count(n);
        offset += 4 + n;

        n = Self::unpack_integer(packing, offset);
        self.s_m3_delta.add_count(n);

        self.s_total.add_count(packing.len() as i32);

        Ok(())
    }

    fn report_analysis_data(&self, mut writer: impl Write, n_tiles_in_raster: i32) -> Result<()> {
        if self.was_data_encoded {
            let avg_bits_per_sample = self.s_total.avg_count() * 8.0 / self.n_cells_in_tile as f64;
            let avg_bytes_per_sample = self.s_total.avg_count() / self.n_cells_in_tile as f64;

            writer.write_str("Gridfour_float\n")?;
            writer.write_str("   Average bytes per tile, by element        (Reduction)\n")?;
            writer.write_fmt(format_args!(
                "     Sign bits           {:.2}        ({:.4}%)\n",
                self.s_sign_bit.avg_count(),
                100.0 * self.s_sign_bit.avg_count() / (self.n_cells_in_tile * 8) as f64
            ))?;
            writer.write_fmt(format_args!(
                "     Exponent            {:.2}        ({:.4}%)\n",
                self.s_exp.avg_count(),
                100.0 * self.s_exp.avg_count() / self.n_cells_in_tile as f64
            ))?;
            writer.write_fmt(format_args!(
                "     Mantissa-1 delta    {:.2}        ({:.4}%)\n",
                self.s_m1_delta.avg_count(),
                100.0 * self.s_m1_delta.avg_count() / (7_f64 * self.n_cells_in_tile as f64 / 8_f64)
            ))?;
            writer.write_fmt(format_args!(
                "     Mantissa-2 delta    {:.2}        ({:.4}%)\n",
                self.s_m2_delta.avg_count(),
                100.0 * self.s_m2_delta.avg_count() / self.n_cells_in_tile as f64
            ))?;
            writer.write_fmt(format_args!(
                "     Mantissa-3 delta    {:.2}        ({:.4}%)\n",
                self.s_m3_delta.avg_count(),
                100.0 * self.s_m3_delta.avg_count() / self.n_cells_in_tile as f64
            ))?;
            writer.write_str("\n")?;
            writer.write_fmt(format_args!(
                "   Average Bytes/Tile   {:.2}        ({:.4}%)\n",
                self.s_total.avg_count(),
                100.0 * self.s_total.avg_count() / (self.n_cells_in_tile * 4) as f64
            ))?;
            writer.write_fmt(format_args!(
                "   Average Bytes/Sample  {:.2}        ({:.4}%)\n",
                avg_bytes_per_sample,
                100.0 * avg_bytes_per_sample / 4_f64
            ))?;

            writer.write_fmt(format_args!(
                "   Average Bits/Sample   {:.2}\n",
                avg_bits_per_sample
            ))?;
        } else {
            writer.write_str("Gridfour_Float (not used)\n")?;
        }

        Ok(())
    }

    fn clear_analysis_data(&mut self) -> Result<()> {
        self.s_sign_bit.clear();
        self.s_exp.clear();
        self.s_m1_delta.clear();
        self.s_m2_delta.clear();
        self.s_m3_delta.clear();
        self.s_total.clear();

        Ok(())
    }

    fn decode_floats(&self, n_rows: i32, n_columns: i32, packing: &[i8]) -> Result<Vec<f32>> {
        todo!()
    }
}

impl CodecFloat {
    fn pack_bytes(output: &mut [i8], offset: i32, sequence: &mut [i8]) -> i32 {
        let sequence_length = sequence.len();
        Self::pack_integer(output, offset, sequence.len() as i32);
        output[(offset + 4) as usize..].copy_from_slice(&sequence[0..sequence_length]);
        offset + sequence_length as i32 + 4
    }

    fn pack_integer(output: &mut [i8], offset: i32, i_value: i32) -> i32 {
        output[offset as usize] = (i_value & MASK as i32) as i8;
        output[(offset + 1) as usize] = ((i_value >> 8) & MASK as i32) as i8;
        output[(offset + 2) as usize] = ((i_value >> 16) & MASK as i32) as i8;
        output[(offset + 3) as usize] = ((i_value >> 24) & MASK as i32) as i8;
        offset + 4
    }

    fn unpack_integer(input: &[i8], offset: i32) -> i32 {
        ((input[offset as usize] & MASK) as i32)
            | ((input[(offset + 1) as usize] & MASK) as i32) << 8
            | ((input[(offset + 2) as usize] & MASK) as i32) << 16
            | ((input[(offset + 2) as usize] & MASK) as i32) << 24
    }

    fn do_deflate(input: &[i8], stats: &mut SimpleStats) -> Result<Vec<i8>> {
        let mut result_b = vec![0; input.len() + 128_usize];
        let result_b_len = result_b.len();
        let mut encoder = GzEncoder::new(result_b, Compression::new(6));
        let db = encoder.write(
            &input[..result_b_len]
                .iter()
                .map(|i| *i as u8)
                .collect::<Vec<_>>(),
        )?;
        encoder.flush()?;
        result_b = encoder.finish()?;
        stats.add_count(db as i32);
        if db == 0 {
            return Err(Error::Compress("deflate failed".to_string()));
        }

        let mut b = vec![0_i8; db];
        b[..result_b_len].copy_from_slice(&result_b.iter().map(|i| *i as i8).collect::<Vec<_>>());
        Ok(b)
    }

    fn do_inflate(
        input: &[i8],
        offset: i32,
        length: i32,
        output: &mut [u8],
        output_length: i32,
    ) -> Result<i32> {
        let mut decoder = GzDecoder::new(output);
        let test = decoder.write(
            &input[..output_length as usize]
                .iter()
                .map(|i| *i as u8)
                .collect::<Vec<_>>(),
        )?;
        if test == 0 {
            return Err(Error::Compress("inflate failed".to_string()));
        }

        Ok(test as i32)
    }

    fn encode_deltas(scratch: &mut [i8], n_rows: i32, n_columns: i32) {
        let mut prior0 = 0_i32;
        let mut test: i32;
        let mut k = 0;
        for _ in 0..n_rows {
            let mut prior = prior0;
            prior0 = scratch[k] as i32;
            for _ in 0..n_columns {
                test = scratch[k] as i32;
                scratch[k] = (test - prior as i32) as i8;
                k += 1;
                prior = test;
            }
        }
    }

    fn decode_deltas(scratch: &mut [i8], n_rows: i32, n_columns: i32) {
        let mut prior = 0_i32;
        let mut k = 0;
        for irow in 0..n_rows {
            for _ in 0..n_columns {
                prior += scratch[k] as i32;
                scratch[k] = prior as i8;
                k += 1;
            }
            prior = scratch[(irow * n_columns) as usize] as i32;
        }
    }
}
struct SimpleStats {
    n_sum: i32,
    sum: i64,
}

impl SimpleStats {
    fn add_count(&mut self, counts: i32) {
        self.sum += counts as i64;
        self.n_sum += 1;
    }

    fn avg_count(&self) -> f64 {
        if self.n_sum == 0 {
            return 0_f64;
        }

        self.sum as f64 / self.n_sum as f64
    }

    fn clear(&mut self) {
        self.n_sum = 0;
        self.sum = 0;
    }
}

#[derive(Default)]
pub struct PredictorModelDifferencing {
    encoded_seed: i32,
}

impl PredictorModel for PredictorModelDifferencing {
    fn decode(
        &self,
        seed: i32,
        n_rows: i32,
        n_columns: i32,
        encoding: &[i8],
        offset: i32,
        length: i32,
        output: &mut [i32],
    ) -> Result<()> {
        let mut m_codec = CodecM32::new(encoding, offset, length);
        output[0] = seed;
        let mut prior = seed;
        for i in 1..n_columns {
            prior += m_codec.decode();
            output[i as usize] = prior;
        }

        for i in 1..n_rows {
            let mut index = i * n_columns;
            prior = output[(index - n_columns) as usize];
            for _ in 0..n_columns {
                prior += m_codec.decode();
                output[index as usize] = prior;
                index += 1;
            }
        }

        Ok(())
    }

    fn encode(
        &mut self,
        n_rows: i32,
        n_columns: i32,
        values: &[i32],
        output: &mut [i8],
    ) -> Result<i32> {
        let mut m_codec = CodecM32::new(output, 0, output.len() as i32);
        self.encoded_seed = values[0];
        let mut prior = self.encoded_seed;
        for i in 1..n_columns {
            let test = values[i as usize];
            let delta = test - prior;
            m_codec.encode(delta);
            prior = test;
        }

        for i in 1..n_rows {
            let mut index = i * n_columns;
            prior = values[(index - n_columns) as usize];
            for _ in 0..n_columns {
                let test = values[index as usize];
                index += 1;
                let delta = test - prior;
                m_codec.encode(delta);
                prior = test;
            }
        }

        output.copy_from_slice(&m_codec.buffer);

        Ok(m_codec.encoded_length())
    }

    fn is_null_data_supported(&self) -> bool {
        false
    }

    fn seed(&self) -> i32 {
        self.encoded_seed
    }

    fn predictor_type(&self) -> PredictorModelType {
        PredictorModelType::Differencing
    }
}

#[derive(Default)]
pub struct PredictorModelLinear {
    encoded_seed: i32,
}

impl PredictorModel for PredictorModelLinear {
    fn decode(
        &self,
        seed: i32,
        n_rows: i32,
        n_columns: i32,
        encoding: &[i8],
        offset: i32,
        length: i32,
        output: &mut [i32],
    ) -> Result<()> {
        let mut m_codec = CodecM32::new(encoding, offset, length);
        let mut prior = seed as i64;
        output[0] = seed;
        output[1] = (m_codec.decode() as i64 + prior) as i32;
        for i in 0..n_rows {
            let index = i * n_columns;
            let test = m_codec.decode() as i64 + prior;
            output[index as usize] = test as i32;
            prior = test;
            output[(index + 1) as usize] = (m_codec.decode() as i64 + test) as i32;
        }

        for i in 0..n_rows {
            let index = i * n_columns;
            let mut a = output[index as usize] as i64;
            let mut b = output[(index + 1) as usize] as i64;

            for j in 2..n_columns {
                let residual = m_codec.decode();
                let prediction = (2_i64 * b - a) as i32;
                let c = prediction + residual;
                a = b;
                b = c as i64;
                output[(index + j) as usize] = c;
            }
        }

        Ok(())
    }

    fn encode(
        &mut self,
        n_rows: i32,
        n_columns: i32,
        values: &[i32],
        encoding: &mut [i8],
    ) -> Result<i32> {
        let mut m_codec = CodecM32::new(encoding, 0, encoding.len() as i32);
        self.encoded_seed = values[0];

        let mut delta = 0_i64;
        let mut test = 0_i64;
        let mut prior = values[0] as i64;
        delta = values[1] as i64 - prior;
        m_codec.encode(delta as i32);
        for i in 1..n_rows {
            let index = i * n_columns;
            test = values[index as usize] as i64;
            delta = test - prior;
            m_codec.encode(delta as i32);
            prior = test;

            test = values[(index + 1) as usize] as i64;
            delta = test - prior;
            m_codec.encode(delta as i32);
        }

        for i in 0..n_rows {
            let index = i * n_columns;
            let mut a = values[index as usize] as i64;
            let mut b = values[(index + 1) as usize] as i64;
            for j in 2..n_columns {
                let c = values[(index + j) as usize];
                let prediction = (2_i64 * b - a) as i32;
                let residual = c - prediction;
                m_codec.encode(residual);
                a = b;
                b = c as i64;
            }
        }

        Ok(m_codec.encoded_length())
    }

    fn is_null_data_supported(&self) -> bool {
        false
    }

    fn seed(&self) -> i32 {
        self.encoded_seed
    }

    fn predictor_type(&self) -> PredictorModelType {
        PredictorModelType::Linear
    }
}

#[derive(Default)]
pub struct PredictorModelTriangle {
    encoded_seed: i32,
}

impl PredictorModel for PredictorModelTriangle {
    fn decode(
        &self,
        seed: i32,
        n_rows: i32,
        n_columns: i32,
        encoding: &[i8],
        offset: i32,
        length: i32,
        output: &mut [i32],
    ) -> Result<()> {
        let mut m_codec = CodecM32::new(encoding, offset, length);

        output[0] = seed;
        let mut prior = seed;
        for i in 1..n_columns {
            prior += m_codec.decode();
            output[i as usize] = prior
        }
        prior = seed;
        for i in 1..n_rows {
            prior += m_codec.decode();
            output[(i * n_columns) as usize] = prior;
        }

        for i in 1..n_rows {
            let mut k1 = i * n_columns;
            let mut k0 = k1 - n_columns;
            for _ in 1..n_columns {
                let za = output[k0 as usize] as i64;
                k0 += 1;
                let zb = output[k1 as usize] as i64;
                k1 += 1;
                let zc = output[k0 as usize] as i64;
                let prediction = (zb + zc - za) as i32;
                output[k1 as usize] = prediction + m_codec.decode();
            }
        }

        Ok(())
    }

    fn encode(
        &mut self,
        n_rows: i32,
        n_columns: i32,
        values: &[i32],
        encoding: &mut [i8],
    ) -> Result<i32> {
        if n_rows < 2 || n_columns < 2 {
            return Ok(-1);
        }

        let mut m_codec = CodecM32::new(encoding, 0, encoding.len() as i32);
        self.encoded_seed = values[0];
        let mut prior = self.encoded_seed as i64;
        for i in 1..n_columns {
            let test = values[i as usize] as i64;
            let delta = test - prior;
            m_codec.encode(delta as i32);
            prior = test;
        }

        prior = self.encoded_seed as i64;
        for i in 1..n_rows {
            let test = values[(i * n_columns) as usize] as i64;
            let delta = test - prior;
            m_codec.encode(delta as i32);
            prior = test;
        }

        for i in 1..n_rows {
            let mut k1 = i * n_columns;
            let mut k0 = k1 - n_columns;
            for _ in 1..n_columns {
                let za = values[k0 as usize] as i64;
                k0 += 1;
                let zb = values[k1 as usize] as i64;
                k1 += 1;
                let zc = values[k0 as usize] as i64;
                let prediction = (zc + zb - za) as i32;
                let residual = values[k1 as usize] - prediction;
                m_codec.encode(residual);
            }
        }

        Ok(m_codec.encoded_length())
    }

    fn is_null_data_supported(&self) -> bool {
        false
    }

    fn seed(&self) -> i32 {
        self.encoded_seed
    }

    fn predictor_type(&self) -> PredictorModelType {
        PredictorModelType::Triangle
    }
}

#[derive(Default)]
pub struct PredictorModelDifferencingWithNulls {
    encoded_seed: i32,
}

impl PredictorModelDifferencingWithNulls {
    const NULL_DATA_CODE: i32 = super::INT4_NULL_CODE;
}

impl PredictorModel for PredictorModelDifferencingWithNulls {
    fn decode(
        &self,
        seed: i32,
        n_rows: i32,
        n_columns: i32,
        encoding: &[i8],
        offset: i32,
        length: i32,
        output: &mut [i32],
    ) -> Result<()> {
        let mut m_codec = CodecM32::new(encoding, offset, length);

        let mut prior = seed;
        let mut null_flag = true;
        for i in 0..n_rows {
            let mut index = i * n_columns;
            for _ in 0..n_columns {
                let test = m_codec.decode();
                if test == PredictorModelDifferencingWithNulls::NULL_DATA_CODE {
                    null_flag = true;
                    output[index as usize] = PredictorModelDifferencingWithNulls::NULL_DATA_CODE;
                    index += 1;
                } else {
                    if null_flag {
                        null_flag = false;
                        prior = seed;
                    }
                    prior += test;
                    output[index as usize] = prior;
                    index += 1;
                }
            }
            prior = output[(i * n_columns) as usize];
            null_flag = prior == PredictorModelDifferencingWithNulls::NULL_DATA_CODE;
        }

        Ok(())
    }

    fn encode(
        &mut self,
        n_rows: i32,
        n_columns: i32,
        values: &[i32],
        output: &mut [i8],
    ) -> Result<i32> {
        let mut m_codec = CodecM32::new(output, 0, output.len() as i32);

        let mut sum_start = 0_i64;
        let mut n_start = 0_i32;
        let mut null_flag = true;
        for i in 0..n_rows {
            let row_offset = i * n_columns;
            for j in 0..n_columns {
                let test = values[(row_offset + j) as usize];
                if test == PredictorModelDifferencingWithNulls::NULL_DATA_CODE {
                    null_flag = true;
                } else {
                    if null_flag {
                        sum_start += test as i64;
                        n_start += 1;
                    }
                    null_flag = false;
                }
            }
            null_flag =
                values[row_offset as usize] == PredictorModelDifferencingWithNulls::NULL_DATA_CODE;
        }

        if n_start == 0 {
            return Ok(0_i32);
        }
        let avg_start = sum_start as f64 / n_start as f64;
        self.encoded_seed = (avg_start + 0.5).floor() as i32;

        let mut prior = self.encoded_seed as i64;
        null_flag = false;
        for i in 0..n_rows {
            let mut index = i * n_columns;
            for _ in 0..n_columns {
                let test = values[index as usize];
                index += 1;
                if test == PredictorModelDifferencingWithNulls::NULL_DATA_CODE {
                    null_flag = true;
                    m_codec.encode(PredictorModelDifferencingWithNulls::NULL_DATA_CODE);
                } else {
                    if null_flag {
                        prior = self.encoded_seed as i64;
                        null_flag = false;
                    }
                    let delta = test as i64 - prior;
                    m_codec.encode(delta as i32);
                    prior = test as i64;
                }
            }
            prior = values[(i * n_columns) as usize] as i64;
            null_flag = prior == PredictorModelDifferencingWithNulls::NULL_DATA_CODE as i64;
        }

        Ok(m_codec.encoded_length())
    }

    fn is_null_data_supported(&self) -> bool {
        true
    }

    fn seed(&self) -> i32 {
        self.encoded_seed
    }

    fn predictor_type(&self) -> PredictorModelType {
        PredictorModelType::DifferencingWithNulls
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod codec_m32 {
        use super::*;

        fn test_size(m_32: &mut CodecM32, input: i32, size: i32) {
            m_32.rewind();
            m_32.encode(input);
            let length = m_32.encoded_length();
            assert_eq!(length, size);
            m_32.rewind();
            let output = m_32.decode();
            assert_eq!(output, input);
        }

        #[test]
        fn single_size() {
            let mut m_32 = CodecM32::from_symbol_count(1);

            for i in -126..126 {
                test_size(&mut m_32, i, 1);
            }

            let _buffer = [0; 6];

            test_size(&mut m_32, 0, 1);
            test_size(&mut m_32, 126, 1);
            test_size(&mut m_32, 127, 2);
            test_size(&mut m_32, -128, 2);
            test_size(&mut m_32, -127, 2);
            test_size(&mut m_32, 128, 2);
            test_size(&mut m_32, -129, 2);
            test_size(&mut m_32, 254, 2);
            test_size(&mut m_32, 255, 3);
            test_size(&mut m_32, 16638, 3);
            test_size(&mut m_32, 16639, 4);
            test_size(&mut m_32, 2113790, 4);
            test_size(&mut m_32, 2113791, 5);
            test_size(&mut m_32, 270549246, 5);
            test_size(&mut m_32, 270549247, 6);
            test_size(&mut m_32, i32::MAX, 6);
            test_size(&mut m_32, i32::MIN + 1, 6);
            test_size(&mut m_32, i32::MIN, 1);
        }
    }

    mod predictor_model_differencing {
        use super::*;

        #[test]
        fn round_trip() {
            let n_rows = 10;
            let n_columns = 10;
            let mut values = vec![0; (n_rows * n_columns) as usize];
            for i_row in 0..n_rows {
                let offset = i_row * n_columns;
                let mut v = i_row;
                for i_col in (0..10).step_by(2) {
                    values[(offset + i_col) as usize] = v;
                    v += 1;
                }
            }

            let mut encoding = vec![0; (n_rows * n_columns * 6) as usize];
            let mut instance = PredictorModelDifferencing::default();

            let encoded_length = instance
                .encode(n_rows, n_columns, &values, &mut encoding)
                .unwrap();
            let seed = instance.seed();

            let mut decoding = vec![0; values.len()];

            instance
                .decode(
                    seed,
                    n_rows,
                    n_columns,
                    &encoding,
                    0,
                    encoded_length,
                    &mut decoding,
                )
                .unwrap();

            for (i, d) in decoding.iter().enumerate() {
                assert_eq!(
                    *d, values[i],
                    "failure to decode at index {}, input={}, output={}",
                    i, values[i], d
                );
            }
        }
    }
}
