use veryl_component::veryl_component_export;

mod analyzer;
mod fm_modulator;
mod generator;
mod utils;

veryl_component_export!("generator" => generator::Generator,
                        "analyzer" => analyzer::Analyzer,
                        "fm_modulator" => fm_modulator::FmModulator,
                        "utils" => utils::Utils);
