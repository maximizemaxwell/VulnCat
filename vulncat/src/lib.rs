pub mod data;
pub mod evaluation;
pub mod inference;
pub mod model;
pub mod preprocessing;
pub mod training;

use anyhow::Result;

pub struct VulnCat {
    model: model::AutoEncoder,
    config: Config,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub model: model::ModelConfig,
    pub training: training::TrainingConfig,
    pub data: data::DataConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: model::ModelConfig::default(),
            training: training::TrainingConfig::default(),
            data: data::DataConfig::default(),
        }
    }
}

impl VulnCat {
    pub fn new(config: Config) -> Result<Self> {
        let model = model::AutoEncoder::new(&config.model)?;
        Ok(Self { model, config })
    }

    pub fn train(&mut self, dataset: &data::CommitDataset) -> Result<()> {
        training::train(&mut self.model, dataset, &self.config.training)
    }

    pub fn detect(&self, commit: &data::Commit) -> Result<f32> {
        inference::detect(&self.model, commit)
    }
}
