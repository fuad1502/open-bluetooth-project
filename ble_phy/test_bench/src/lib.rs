use veryl_component::veryl_component_export;

mod analyzer;
mod generator;

veryl_component_export!("generator" => generator::Generator, "analyzer" => analyzer::Analyzer);
