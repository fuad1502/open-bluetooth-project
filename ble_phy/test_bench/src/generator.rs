use std::f64::consts::PI;

use veryl_component::*;

#[derive(Component)]
#[component(kind = clocked)]
pub struct Generator {
    #[param(name = "Width")]
    width: u64,
    #[param(name = "Freq")]
    freq: u64,
    #[param(name = "SampleRate")]
    sample_rate: u64,

    clk: ClockPort,
    rst: ResetPort,
    data: OutputPort,

    n: u64,
}

#[component_impl]
impl Generator {
    fn on_clock(&mut self, ctx: &mut SimCtx) -> Result<()> {
        self.n += 1;
        let value = Self::calculate_value(
            self.freq as f64,
            self.sample_rate as f64,
            self.n,
            self.width,
        );
        ctx.write(self.data, value);
        Ok(())
    }

    fn on_reset(&mut self, ctx: &mut SimCtx) -> Result<()> {
        self.n = 0;
        ctx.write(self.data, 0);
        Ok(())
    }
}

impl Generator {
    fn calculate_value(freq: f64, sample_rate: f64, n: u64, width: u64) -> i64 {
        let tetha = 2.0 * PI * freq / sample_rate * n as f64;
        let value = tetha.sin();
        let scaled_value = value * (((width - 1) as f64).exp2() - 1 as f64);
        scaled_value as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veryl_component::testing::MockSim;

    #[test]
    fn clock_runs() {
        let freq = 1e6;
        let sample_rate = 4e6;
        let width = 32;

        let mut sim = MockSim::new()
            .param("Width", width)
            .param("Freq", freq as u64)
            .param("SampleRate", sample_rate as u64)
            .clock_port("clk")
            .reset_port("rst")
            .output("data", width);
        let mut c = sim.build::<Generator>().unwrap();

        sim.reset(&mut c).unwrap();
        let data = sim.get("data").as_i64().unwrap();
        let expected_data = 0;
        assert!(data == expected_data);

        sim.clock(&mut c).unwrap();
        let data = sim.get("data").as_i64().unwrap();
        let expected_data = (1 << 31) - 1;
        assert!(data == expected_data);

        sim.clock(&mut c).unwrap();
        let data = sim.get("data").as_i64().unwrap();
        let expected_data = 0;
        assert!(data == expected_data);

        sim.clock(&mut c).unwrap();
        let data = sim.get("data").as_i64().unwrap();
        let expected_data = -((1 << 31) - 1);
        assert!(data == expected_data);

        sim.reset(&mut c).unwrap();
        sim.clock(&mut c).unwrap();
        let data = sim.get("data").as_i64().unwrap();
        let expected_data = (1 << 31) - 1;
        assert!(data == expected_data);

        assert!(!sim.failed());
    }
}
