use super::{codec::CodecM32, *};

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
        for i in 1..n_rows {
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

        let mut delta: i64;
        let mut test: i64;
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

        encoding.copy_from_slice(&m_codec.buffer);

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

        encoding.copy_from_slice(&m_codec.buffer);

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

        output.copy_from_slice(&m_codec.buffer);

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

    mod predictor_model_differencing_with_nulls {
        use super::*;

        #[test]
        fn round_trip() {
            let n_rows = 10;
            let n_columns = 10;
            let mut values = vec![0_i32; (n_rows * n_columns) as usize];
            for i_row in 0..n_rows {
                let offset = i_row * n_columns;
                let mut v = i_row;
                for i_col in (0..10).step_by(2) {
                    values[(offset + i_col) as usize] = v;
                    v += 1;
                }
                values[(offset + i_row) as usize] = i32::MIN;
            }

            let mut encoding = vec![0; (n_rows * n_columns * 6) as usize];
            let mut instance = PredictorModelDifferencingWithNulls::default();

            assert!(
                instance.is_null_data_supported(),
                "implementation does not support null data"
            );

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

    mod predictor_model_linear {
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
            let mut instance = PredictorModelLinear::default();

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

    mod predictor_model_triangle {
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
            let mut instance = PredictorModelTriangle::default();

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
