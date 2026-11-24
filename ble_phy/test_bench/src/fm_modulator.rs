use std::f64::consts::PI;

use veryl_component::*;

#[derive(Component)]
#[component(kind = clocked)]
pub struct FmModulator {
    #[param(name = "Width")]
    width: u32,
    #[param(name = "SampleRate")]
    sample_rate: u64,

    #[allow(dead_code)]
    clk: ClockPort,
    #[allow(dead_code)]
    rst: ResetPort,
    freq: InputPort,
    o_i: OutputPort,
    o_q: OutputPort,

    n: u64,
}

#[component_impl]
impl FmModulator {
    fn on_clock(&mut self, ctx: &mut SimCtx) -> Result<()> {
        self.n += 1;
        let cosine = Self::cosine(
            self.freq(ctx),
            0.0,
            self.sample_rate as f64,
            self.n,
            self.width,
        );
        let sine = Self::cosine(
            self.freq(ctx),
            PI / 2.0,
            self.sample_rate as f64,
            self.n,
            self.width,
        );
        ctx.write(self.o_i, cosine);
        ctx.write(self.o_q, sine);
        Ok(())
    }

    fn on_reset(&mut self, ctx: &mut SimCtx) -> Result<()> {
        self.n = 0;
        ctx.write(self.o_i, 0);
        ctx.write(self.o_q, 0);
        Ok(())
    }
}

impl FmModulator {
    pub fn cosine(freq: f64, delay: f64, sample_rate: f64, n: u64, width: u32) -> i64 {
        let tetha = 2.0 * PI * freq / sample_rate * n as f64 - delay;
        let value = tetha.cos();
        let scaled_value = value * (((width - 1) as f64).exp2() - 1 as f64);
        scaled_value as i64
    }

    fn freq(&self, ctx: &mut SimCtx) -> f64 {
        ctx.read(self.freq).as_u64().unwrap() as f64
    }
}
