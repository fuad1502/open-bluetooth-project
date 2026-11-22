use veryl_component::*;

#[derive(Component)]
#[component(kind = clocked)]
pub struct Analyzer {
    clk: ClockPort,
    rst: ResetPort,
    data: InputPort,
}

#[component_impl]
impl Analyzer {
    fn on_clock(&mut self, ctx: &mut SimCtx) -> Result<()> {
        Ok(())
    }

    fn on_reset(&mut self, ctx: &mut SimCtx) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veryl_component::testing::MockSim;

    #[test]
    fn clock_runs() {}
}
