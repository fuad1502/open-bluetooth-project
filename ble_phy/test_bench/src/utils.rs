use veryl_component::*;

// TODO: Remove this component! Change to Atan verification model.
#[derive(Component)]
#[component(kind = method_only)]
pub struct Utils {
    #[param(name = "Width")]
    width: u32,
}

#[component_impl]
impl Utils {
    #[ret_width(Width)]
    fn convert_degree(&mut self, _ctx: &mut SimCtx, degree: u64) -> Result<Value> {
        let degree = degree as f64 / ((1i64 << (self.width - 1)) - 1) as f64 * 0.48 * 180.0;
        Ok(Value::from_u64(degree as u64, self.width))
    }
}
