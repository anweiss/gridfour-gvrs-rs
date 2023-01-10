use std::fmt::Display;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("io error {0}")]
    Io(String),
}

pub struct BitOutputStore {
    mask: [i64; 65],
    marker: i64,
    scratch: i64,
    n_bits: i32,
    byte_buffer: ByteBuffer,
}

impl BitOutputStore {
    const BLOCK_SIZE: i32 = 1024;
}

impl Default for BitOutputStore {
    fn default() -> Self {
        let mut mask = [0; 65];

        let mut m = 1_i64;
        for i in mask.iter_mut().take(64).skip(1) {
            *i = m;
            m = (m << 1) | 1_i64;
        }

        BitOutputStore {
            mask,
            marker: 1,
            scratch: 0,
            n_bits: 0,
            byte_buffer: ByteBuffer::default(),
        }
    }
}

impl Display for BitOutputStore {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "BitOutputStore nBits={}", self.n_bits)
    }
}

impl BitOutputStore {
    pub fn append_bit(&mut self, value: i32) {
        if value != 0 {
            self.scratch |= self.marker;
        }
        self.marker <<= 1;
        self.n_bits += 1;
        if self.marker == 0 {
            self.move_scratch_to_text();
        }
    }

    pub fn append_bits(&mut self, n_bits_in_value: i32, value: i32) -> Result<()> {
        if !(1..=32).contains(&n_bits_in_value) {
            return Err(Error::Io(format!(
                "attempt to add number of bits no in range (1, 32): {}",
                n_bits_in_value
            )));
        }

        let v = value as i64 & self.mask[n_bits_in_value as usize];

        let n_bits_in_scratch = self.n_bits & 0x3F;
        let n_free_in_scratch = 64 - n_bits_in_scratch;

        if n_free_in_scratch < n_bits_in_value {
            let n_bits_short = n_bits_in_value - n_free_in_scratch;
            let low_part = v & self.mask[n_free_in_scratch as usize];
            let v_as_u64: u64 = {
                let bytes = v.to_be_bytes();
                u64::from_be_bytes(bytes)
            };
            let high_part = (v_as_u64 >> n_free_in_scratch) as i64;
            self.scratch |= low_part << n_bits_in_scratch;
            self.n_bits += n_free_in_scratch;
            self.move_scratch_to_text();
            self.scratch = high_part;
            self.n_bits += n_bits_short;
            self.marker = 1_i64 << n_bits_short;
        } else {
            self.scratch |= v << n_bits_in_scratch;
            self.n_bits += n_bits_in_value;
            self.marker <<= n_bits_in_value;
            if self.marker == 0 {
                self.move_scratch_to_text();
            }
        }

        Ok(())
    }

    pub fn encoded_text(&self) -> Vec<i8> {
        let n_bytes_to_encode = (self.n_bits + 7) / 8;
        let mut b = self.byte_buffer.bytes(n_bytes_to_encode);

        let n_bits_in_scratch = self.n_bits & 0x3F;
        if n_bits_in_scratch > 0 {
            let mut s = self.scratch;
            let n_bytes_in_scratch = (n_bits_in_scratch + 7) / 8;
            let mut i_byte = self.byte_buffer.byte_count();
            for _ in 0..n_bytes_in_scratch {
                b[i_byte as usize] = (s & 0xff_i64) as i8;
                i_byte += 1;
                s >>= 8;
            }
        }

        b
    }

    pub fn encoded_text_length(&self) -> i32 {
        self.n_bits
    }

    pub fn encoded_text_length_in_bytes(&self) -> i32 {
        (self.encoded_text_length() + 7) / 8
    }

    fn move_scratch_to_text(&mut self) {
        self.byte_buffer.add_long(self.scratch);
        self.scratch = 0;
        self.marker = 1;
    }
}

struct ByteBuffer {
    i_byte: i32,
    // block: [i8; BitOutputStore::BLOCK_SIZE as usize],
    block_list: Vec<[i8; BitOutputStore::BLOCK_SIZE as usize]>,
    block_idx: usize,
}

impl Default for ByteBuffer {
    fn default() -> Self {
        let mut block_list = Vec::default();
        let block = [0; BitOutputStore::BLOCK_SIZE as usize];
        block_list.push(block);

        ByteBuffer {
            block_list,
            i_byte: 0,
            block_idx: 0,
        }
    }
}

impl ByteBuffer {
    fn add_byte(&mut self, b: i8) {
        if self.i_byte == BitOutputStore::BLOCK_SIZE {
            self.i_byte = 0;
            self.block_list
                .push([0; BitOutputStore::BLOCK_SIZE as usize]);
            self.block_idx += 1;
        }
        if let Some(block) = self.block_list.get_mut(self.block_idx) {
            block[self.i_byte as usize] = b;
        }
        self.i_byte += 1;
    }

    fn add_long(&mut self, value: i64) {
        let mut s = value;
        self.add_byte((s & 0xff) as i8);
        s >>= 8;
        self.add_byte((s & 0xff) as i8);
        s >>= 8;
        self.add_byte((s & 0xff) as i8);
        s >>= 8;
        self.add_byte((s & 0xff) as i8);
        s >>= 8;
        self.add_byte((s & 0xff) as i8);
        s >>= 8;
        self.add_byte((s & 0xff) as i8);
        s >>= 8;
        self.add_byte((s & 0xff) as i8);
        s >>= 8;
        self.add_byte((s & 0xff) as i8);
    }

    fn byte_count(&self) -> i32 {
        (self.block_list.len() - 1) as i32 * BitOutputStore::BLOCK_SIZE + self.i_byte
    }

    fn bytes(&self, n_bytes_required: i32) -> Vec<i8> {
        let n = self.byte_count();
        let mut n_alloc = n;
        if n_bytes_required > n {
            n_alloc = n_bytes_required;
        }

        let mut b = vec![0; n_alloc as usize];
        if n == 0 {
            return b;
        }

        let n_full_blocks = n / BitOutputStore::BLOCK_SIZE;
        for i in 0..n_full_blocks {
            if let Some(s) = self.block_list.get(i as usize) {
                let dst_start_pos = i * BitOutputStore::BLOCK_SIZE;
                let dst_end_pos = dst_start_pos + BitOutputStore::BLOCK_SIZE;
                b[dst_start_pos as usize..dst_end_pos as usize]
                    .copy_from_slice(&s[..BitOutputStore::BLOCK_SIZE as usize]);
            }
        }
        let b_offset = n_full_blocks * BitOutputStore::BLOCK_SIZE;
        if b_offset < n && n_full_blocks < self.block_list.len() as i32 {
            let dst_start_pos = b_offset;
            let dst_end_pos = dst_start_pos + (n - b_offset);
            if let Some(block) = self.block_list.get(self.block_idx) {
                b[dst_start_pos as usize..dst_end_pos as usize]
                    .copy_from_slice(&block[..(n - b_offset) as usize]);
            }
        }

        b
    }
}

pub struct BitInputStore {
    mask: [i64; 65],
    text: Vec<i8>,
    n_bits: i32,
    n_bytes_processed: i32,
    scratch: i64,
    i_bit: i32,
    n_bits_in_scratch: i32,
}

impl BitInputStore {
    pub fn new(input: &[i8]) -> Self {
        let mut mask = [0; 65];

        let mut m = 1_i64;
        for i in mask.iter_mut().take(64).skip(1) {
            *i = m;
            m = (m << 1) | 1_i64;
        }

        BitInputStore {
            mask,
            text: input.to_owned(),
            n_bits: (input.len() * 8) as i32,
            scratch: 0,
            i_bit: 0,
            n_bits_in_scratch: 0,
            n_bytes_processed: 0,
        }
    }

    pub fn from_input_at_offset(input: &[i8], offset: i32, length: i32) -> Result<Self> {
        if length + offset > input.len() as i32 {
            return Err(Error::Io(format!(
                "insufficient input.length={} to support specified offset={}, length={}",
                input.len(),
                offset,
                length
            )));
        }
        let mut mask = [0; 65];

        let mut m = 1_i64;
        for i in mask.iter_mut().take(64).skip(1) {
            *i = m;
            m = (m << 1) | 1_i64;
        }

        Ok(BitInputStore {
            mask,
            n_bits: length * 8,
            n_bits_in_scratch: 0,
            text: input.to_owned(),
            n_bytes_processed: offset,
            scratch: 0,
            i_bit: 0,
        })
    }

    pub fn bit(&mut self) -> Result<i32> {
        if self.n_bits_in_scratch == 0 {
            if self.i_bit >= self.n_bits {
                return Err(Error::Io("attempt to read past end of data".to_string()));
            }

            self.move_text_to_scratch();
        }

        let bit = (self.scratch & 1_i64) as i32;

        let scratch_as_u64: u64 = {
            let bytes = self.scratch.to_be_bytes();
            u64::from_be_bytes(bytes)
        };
        self.scratch = (scratch_as_u64 >> 1) as i64;
        self.n_bits_in_scratch -= 1;
        self.i_bit += 1;

        Ok(bit)
    }

    pub fn bits(&mut self, n_bits_in_value: i32) -> Result<i32> {
        if !(1..=32).contains(&n_bits_in_value) {
            return Err(Error::Io(format!(
                "attempt to get a number of bits not in range [1..32]: {}",
                n_bits_in_value
            )));
        }

        if self.n_bits_in_scratch >= n_bits_in_value {
            let v = (self.scratch & self.mask[n_bits_in_value as usize]) as i32;
            let scratch_as_u64: u64 = {
                let bytes = self.scratch.to_be_bytes();
                u64::from_be_bytes(bytes)
            };
            self.scratch = (scratch_as_u64 >> n_bits_in_value) as i64;
            self.n_bits_in_scratch -= n_bits_in_value;
            self.i_bit += n_bits_in_value;
            return Ok(v);
        }

        if self.i_bit + n_bits_in_value > self.n_bits {
            return Err(Error::Io("attempt to read past end of data".to_string()));
        }

        let mut v = self.scratch;

        let n_bits_short = n_bits_in_value - self.n_bits_in_scratch;
        let n_bits_copied = self.n_bits_in_scratch;
        self.move_text_to_scratch();
        v |= (self.scratch & self.mask[n_bits_short as usize]) << n_bits_copied;
        let scratch_as_u64: u64 = {
            let bytes = self.scratch.to_be_bytes();
            u64::from_be_bytes(bytes)
        };
        self.scratch = (scratch_as_u64 >> n_bits_short) as i64;
        self.n_bits_in_scratch -= n_bits_short;
        self.i_bit += n_bits_in_value;

        Ok(v as i32)
    }

    fn move_text_to_scratch(&mut self) {
        if self.n_bytes_processed + 8 <= self.text.len() as i32 {
            self.scratch = (((((((self.text[(self.n_bytes_processed + 7) as usize] as i64)
                << 8
                | (self.text[(self.n_bytes_processed + 6) as usize] as i64 & 0xff_i64))
                << 8
                | (self.text[(self.n_bytes_processed + 5) as usize] as i64 & 0xff_i64))
                << 8
                | (self.text[(self.n_bytes_processed + 4) as usize] as i64 & 0xff_i64))
                << 8
                | (self.text[(self.n_bytes_processed + 3) as usize] as i64 & 0xff_i64))
                << 8
                | (self.text[(self.n_bytes_processed + 2) as usize] as i64 & 0xff_i64))
                << 8
                | (self.text[(self.n_bytes_processed + 1) as usize] as i64 & 0xff_i64))
                << 8
                | (self.text[(self.n_bytes_processed) as usize] as i64 & 0xff_i64);

            self.n_bytes_processed += 8;
            self.n_bits_in_scratch = 64;
        } else {
            let mut k = 0_i32;
            self.scratch = 0;
            for i in (self.n_bytes_processed..self.text.len() as i32).rev() {
                self.scratch <<= 8;
                self.scratch |= self.text[i as usize] as i64 & 0xff;
                k += 1;
            }
            self.n_bytes_processed += k;
            self.n_bits_in_scratch = k * 8;
        }
    }

    pub fn position(&self) -> i32 {
        self.i_bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rand_distr::{Distribution, Uniform};

    fn mask() -> [i32; 33] {
        let mut mask = [0; 33];

        let mut m = 1;
        for i in 1..mask.len() {
            mask[i] = m;
            m = (m << 1) | 1;
        }

        mask
    }

    #[test]
    fn round_trip() {
        let n_test = 5;
        for i in 0..n_test {
            test(i, 1000000);
        }
    }

    fn test(seed: i32, n_symbols_in_text: i32) {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut bit_count = 0;

        let mut n_bits = vec![0; n_symbols_in_text as usize];
        let mut values = vec![0; n_symbols_in_text as usize];

        let mut writer = BitOutputStore::default();
        for i in 0..n_symbols_in_text {
            let between = Uniform::from(0..32);
            let n: i32 = between.sample(&mut rng) + 1;
            let v = rng.gen();
            n_bits[i as usize] = n;
            values[i as usize] = v & mask()[n as usize];
            writer.append_bits(n, v).unwrap();
            writer.append_bit(i & 1);
            bit_count += n + 1;
        }

        let n_bits_in_text = writer.encoded_text_length();
        assert_eq!(n_bits_in_text, bit_count, "encoded bit count mismatch");

        let n_bytes_in_text = (n_bits_in_text + 7) / 8;

        let content = writer.encoded_text();
        assert_eq!(
            content.len(),
            n_bytes_in_text as usize,
            "encoding length mismatch"
        );

        let mut reader = BitInputStore::new(&content);
        for i in 0..n_symbols_in_text {
            let n = n_bits[i as usize];
            let v0 = values[i as usize];
            let v1 = reader.bits(n).unwrap();

            assert_eq!(v1, v0, "mismatch values");

            let a = reader.bit().unwrap();
            assert_eq!(i & 1, a, "mismatch single-bit test");
        }
    }
}
