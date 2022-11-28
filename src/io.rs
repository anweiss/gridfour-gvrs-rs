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
            m = (m << 1) as i64 | 1_i64;
        }

        BitOutputStore {
            mask,
            marker: 1,
            ..Default::default()
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
            self.scratch |= low_part << n_free_in_scratch;
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

    fn move_scratch_to_text(&mut self) {
        self.byte_buffer.add_long(self.scratch);
        self.scratch = 0;
        self.marker = 1;
    }
}

struct ByteBuffer {
    i_byte: i32,
    block: [i8; BitOutputStore::BLOCK_SIZE as usize],
    block_list: Vec<[i8; BitOutputStore::BLOCK_SIZE as usize]>,
}

impl Default for ByteBuffer {
    fn default() -> Self {
        let mut block_list = Vec::default();
        let block = [0; BitOutputStore::BLOCK_SIZE as usize];
        block_list.push(block);

        ByteBuffer {
            block_list,
            block,
            i_byte: 0,
        }
    }
}

impl ByteBuffer {
    fn add_byte(&mut self, b: i8) {
        if self.i_byte == BitOutputStore::BLOCK_SIZE {
            self.i_byte = 0;
            self.block = [0; BitOutputStore::BLOCK_SIZE as usize];
            self.block_list.push(self.block);
        }
        self.block[self.i_byte as usize] = b;
        self.i_byte += 1;
    }

    fn add_long(&mut self, value: i64) {
        let mut s = value;
        self.add_byte((s & 0xff_i64) as i8);
        s >>= 8;
        self.add_byte((s & 0xff_i64) as i8);
        s >>= 8;
        self.add_byte((s & 0xff_i64) as i8);
        s >>= 8;
        self.add_byte((s & 0xff_i64) as i8);
        s >>= 8;
        self.add_byte((s & 0xff_i64) as i8);
        s >>= 8;
        self.add_byte((s & 0xff_i64) as i8);
        s >>= 8;
        self.add_byte((s & 0xff_i64) as i8);
        s >>= 8;
        self.add_byte((s & 0xff_i64) as i8);
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
                b[(i * BitOutputStore::BLOCK_SIZE) as usize..]
                    .copy_from_slice(&s[..BitOutputStore::BLOCK_SIZE as usize]);
            }
        }
        let b_offset = n_full_blocks * BitOutputStore::BLOCK_SIZE;
        if b_offset < n && n_full_blocks < self.block_list.len() as i32 {
            b[b_offset as usize..].copy_from_slice(&self.block[..(n - b_offset) as usize]);
        }

        b
    }
}
