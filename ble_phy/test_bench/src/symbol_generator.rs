use std::f64::consts::{LN_2, PI};

use veryl_component::*;

#[derive(Component)]
#[component(kind = clocked)]
pub struct SymbolGenerator {
    #[param(name = "Width")]
    width: u32,
    #[param(name = "Fractions")]
    fractions: u32,
    #[param(name = "SamplesPerSymbol")]
    sps: u32,
    #[param(name = "BT")]
    bt: u32,
    #[param(name = "SampleOffset")]
    sample_offset: u32,

    #[allow(dead_code)]
    clk: ClockPort,
    #[allow(dead_code)]
    rst: ResetPort,
    i_amplitude: InputPort,
    i_offset: InputPort,
    i_noise_db: InputPort,
    i_data: InputPort,
    i_data_en: InputPort,
    o_data: OutputPort,

    data_buffer: Vec<u64>,
    output_buffer: Vec<i64>,
    impulse_response: Vec<f64>,
    sample: usize,
}

#[component_impl]
impl SymbolGenerator {
    fn on_build(&mut self, ctx: &mut BuildCtx) -> Result<()> {
        if ctx.input("i_amplitude").unwrap().width() != self.width {
            return Err(Error::msg("Port size mismatch"));
        }
        if ctx.input("i_offset").unwrap().width() != self.width {
            return Err(Error::msg("Port size mismatch"));
        }
        if ctx.input("i_noise_db").unwrap().width() != self.width {
            return Err(Error::msg("Port size mismatch"));
        }
        if ctx.input("i_data").unwrap().width() != 64 {
            return Err(Error::msg("Port size mismatch"));
        }
        if ctx.output("o_data").unwrap().width() != self.width {
            return Err(Error::msg("Port size mismatch"));
        }

        let buffer_size = self.sps as usize * 3 + (if self.sps % 2 == 0 { 1 } else { 0 });
        self.output_buffer.resize(buffer_size, -1);
        self.impulse_response.resize(buffer_size, 0.0);

        Ok(())
    }

    fn on_clock(&mut self, ctx: &mut SimCtx) -> Result<()> {
        let o_data = self.calc_symbol(ctx);
        let o_noise = self.calc_noise(ctx);
        let o_total = o_data + o_noise;
        ctx.write_u64(self.o_data, o_total as u64);

        let i_data_en = ctx.read(self.i_data_en).as_bool();
        let i_data = ctx.read_u64(self.i_data);
        if i_data_en {
            self.data_buffer.push(i_data);
        }

        Ok(())
    }

    fn on_reset(&mut self, ctx: &mut SimCtx) -> Result<()> {
        self.data_buffer.clear();
        self.sample = 0;
        ctx.write_u64(self.o_data, 0);
        Ok(())
    }
}

impl SymbolGenerator {
    fn calc_symbol(&mut self, ctx: &mut SimCtx) -> i64 {
        let symbol = if let Some(data) = self.data_buffer.first() {
            let bit = (data >> (self.sample / self.sps as usize) & 0b1) as i64;
            self.sample += 1;
            bit * 2 - 1
        } else {
            -1
        };

        if self.sample >= (64 * self.sps) as usize {
            self.sample = 0;
            self.data_buffer.remove(0);
        }

        let offset = ctx.read_u64(self.i_offset) as i64;
        let amplitude = ctx.read_u64(self.i_amplitude) as i64;

        self.output_buffer.insert(0, symbol);
        self.output_buffer.pop();

        self.recalc_impulse_response(ctx);

        self.output_buffer
            .iter()
            .map(|x| x * amplitude + offset)
            .zip(&self.impulse_response)
            .map(|(x, h)| x as f64 * h)
            .sum::<f64>() as i64
    }

    fn calc_noise(&self, ctx: &mut SimCtx) -> i64 {
        let amplitude = ctx.read_u64(self.i_amplitude) as i64;
        let noise_db = ctx.read_u64(self.i_noise_db) as i64;
        let loss = 10f64.powf((-noise_db) as f64 / 20.0);
        let noise_amplitude = (amplitude as f64 * loss) as i64;

        if noise_amplitude == 0 {
            0
        } else {
            rand::random_range(-noise_amplitude..noise_amplitude)
        }
    }

    fn recalc_impulse_response(&mut self, _ctx: &mut SimCtx) {
        let buffer_size = self.impulse_response.len();
        let bt = self.bt as i32 as f64 / (1 << self.fractions) as f64;
        let sample_offset = self.sample_offset as i32 as f64 / (1 << self.fractions) as f64;

        let sigma = f64::sqrt(LN_2) / (2.0 * PI * bt) * self.sps as f64;
        for i in 0..buffer_size {
            let n = i as f64 - buffer_size as f64 / 2.0 + 0.5 + sample_offset;
            self.impulse_response[i] = f64::exp(-1.0 * n.powi(2) / (2.0 * sigma.powi(2)));
        }

        let scale: f64 = self.impulse_response.iter().sum();
        self.impulse_response
            .iter_mut()
            .for_each(|x| *x = *x / scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veryl_component::testing::MockSim;

    #[test]
    fn test() {
        let width = 32;
        let fractions = 30;
        let sps = 4;
        let bt = (0.5 * (1 << fractions) as f64) as i32 as u32;
        let sample_offset = (0.25 * (1 << fractions) as f64) as i32 as u32;
        let amplitude = 50;
        let offset = 10;
        let data = 0b11110101;

        let mut sim = MockSim::new()
            .param("Width", width)
            .param("Fractions", fractions)
            .param("SamplesPerSymbol", sps)
            .param("BT", bt)
            .param("SampleOffset", sample_offset)
            .clock_port("clk")
            .reset_port("rst")
            .input("i_amplitude", width)
            .input("i_offset", width)
            .input("i_noise_db", width)
            .input("i_data", 64)
            .input("i_data_en", 1)
            .output("o_data", width);
        let mut c = sim.build::<SymbolGenerator>().unwrap();

        sim.reset(&mut c).unwrap();
        sim.set("i_amplitude", amplitude);
        sim.set("i_offset", offset);
        sim.set("i_noise_db", 100);
        sim.set("i_data", data);
        sim.set("i_data_en", 1);
        sim.clock(&mut c).unwrap();
        sim.set("i_data_en", 0);

        for _ in 0..(sps * 3 / 2) {
            sim.clock(&mut c).unwrap();
        }

        for i in 0..8 {
            let expected = (data >> i) & 0b1;

            sim.clock(&mut c).unwrap();

            sim.clock(&mut c).unwrap();
            let data = sim.get("o_data").as_i64().unwrap();
            if expected == 1 {
                assert!(data > offset);
            } else {
                assert!(data < offset);
            }

            sim.clock(&mut c).unwrap();
            let data = sim.get("o_data").as_i64().unwrap();
            if expected == 1 {
                assert!(data > offset);
            } else {
                assert!(data < offset);
            }

            sim.clock(&mut c).unwrap();
        }
    }
}
