use std::f64::consts::PI;

use rustfft::{self, FftPlanner, num_complex::Complex};
use veryl_component::*;

#[derive(Component)]
#[component(kind = clocked)]
pub struct Analyzer {
    #[param(name = "Width")]
    width: u32,
    #[param(name = "SampleRate")]
    sample_rate: u64,

    #[allow(dead_code)]
    clk: ClockPort,
    #[allow(dead_code)]
    rst: ResetPort,
    data: InputPort,
    clk_en: InputPort,

    buffer: Vec<Complex<f64>>,
}

#[component_impl]
impl Analyzer {
    fn on_clock(&mut self, ctx: &mut SimCtx) -> Result<()> {
        if ctx.read(self.clk_en).as_bool() {
            let data = ctx.read(self.data).as_i64().unwrap();
            let normalized = (data as f64) / ((self.width - 1) as f64).exp2();
            self.buffer.push(Complex {
                re: normalized,
                im: 0f64,
            });
        }
        Ok(())
    }

    fn on_reset(&mut self, _ctx: &mut SimCtx) -> Result<()> {
        self.buffer.clear();
        Ok(())
    }

    fn check_loss(
        &mut self,
        ctx: &mut SimCtx,
        freq: u64,
        expected_loss_db: u64,
        tolerance_db: u64,
    ) -> Result<()> {
        let expected_db: i64 = -(expected_loss_db as i64);
        let mag_db = 20.0 * self.calc_fft(freq as f64).log10();
        if f64::abs(mag_db - expected_db as f64) > tolerance_db as f64 {
            ctx.fail(format!(
                "mag_db = {mag_db}, expected_db = {expected_db}, tolerance_db = {tolerance_db}"
            ));
        }
        Ok(())
    }

    fn check_loss_greater_than(
        &mut self,
        ctx: &mut SimCtx,
        freq: u64,
        min_loss_db: u64,
    ) -> Result<()> {
        let max_db: i64 = -(min_loss_db as i64);
        let mag_db = 20.0 * self.calc_fft(freq as f64).log10();
        if mag_db > max_db as f64 {
            ctx.fail(format!("mag_db = {mag_db}, max_db = {max_db}"));
        }
        Ok(())
    }

    fn check_loss_less_than(
        &mut self,
        ctx: &mut SimCtx,
        freq: u64,
        max_loss_db: u64,
    ) -> Result<()> {
        let min_db: i64 = -(max_loss_db as i64);
        let mag_db = 20.0 * self.calc_fft(freq as f64).log10();
        if mag_db < min_db as f64 {
            ctx.fail(format!("mag_db = {mag_db}, min_db = {min_db}"));
        }
        Ok(())
    }
}

impl Analyzer {
    fn calc_fft(&self, freq: f64) -> f64 {
        let mut planner = FftPlanner::<f64>::new();
        let mut buffer = self.hamming();
        let fft = planner.plan_fft_forward(buffer.len());
        fft.process(&mut buffer);
        let bin = self.calc_bin(freq);
        Self::calc_lobe_mag(bin, &buffer)
    }

    fn hamming(&self) -> Vec<Complex<f64>> {
        let length = self.buffer.len();
        self.buffer
            .iter()
            .enumerate()
            .map(|(i, e)| e * (0.54 - 0.46 * f64::cos(2.0 * PI * i as f64 / length as f64)))
            .collect()
    }

    fn calc_bin(&self, freq: f64) -> usize {
        (freq as f64 * self.buffer.len() as f64 / self.sample_rate as f64) as usize
    }

    fn calc_lobe_mag(bin: usize, bins: &Vec<Complex<f64>>) -> f64 {
        let mut lobe_magnitude = 0f64;
        let bin = bin as i64;
        for i in -2..=2 {
            let bin = (bin + i).rem_euclid(bins.len() as i64);
            lobe_magnitude += bins[bin as usize].norm() / bins.len() as f64 * 2.0;
        }
        lobe_magnitude
    }
}

#[cfg(test)]
mod tests {
    use veryl_component::{Value, testing::MockSim};

    use crate::{analyzer::Analyzer, generator::Generator};

    #[test]
    fn sine() {
        let width = 32;
        let sample_rate = 4000000;
        let freq = 100000;

        let mut sim = MockSim::new()
            .param("Width", width)
            .param("SampleRate", sample_rate)
            .clock_port("clk")
            .reset_port("rst")
            .input("clk_en", 1)
            .input("data", width);
        let mut c = sim.build::<Analyzer>().unwrap();

        sim.reset(&mut c).unwrap();
        for i in 0..1024 {
            let value = Generator::sine(freq as f64, sample_rate as f64, i, width);
            sim.set("data", value);
            sim.set("clk_en", 1);
            sim.clock(&mut c).unwrap();
            sim.set("clk_en", 0);
            sim.clock(&mut c).unwrap();
            sim.clock(&mut c).unwrap();
            sim.clock(&mut c).unwrap();
            sim.clock(&mut c).unwrap();
        }

        let freq = Value::from_u64(freq, width);
        let expected_loss_db = Value::from_u64(0, width);
        let tolerance_db = Value::from_u64(1, width);
        sim.call(
            &mut c,
            "check_loss",
            &[freq, expected_loss_db, tolerance_db],
        )
        .unwrap();

        sim.reset(&mut c).unwrap();
        assert!(!sim.failed());
    }
}
