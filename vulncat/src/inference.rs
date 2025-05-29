use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::{
    data::Commit,
    model::AutoEncoder,
    preprocessing::{PreprocessingConfig, Preprocessor},
};

pub struct VulnerabilityDetector {
    threshold_percentile: f32,
    baseline_error: f32,
    max_error: f32,
}

impl Default for VulnerabilityDetector {
    fn default() -> Self {
        Self {
            threshold_percentile: 0.95,
            baseline_error: 0.1,
            max_error: 10.0,
        }
    }
}

impl VulnerabilityDetector {
    pub fn new(baseline_error: f32, max_error: f32) -> Self {
        Self {
            threshold_percentile: 0.95,
            baseline_error,
            max_error,
        }
    }

    pub fn calculate_score(&self, reconstruction_error: f32) -> f32 {
        let normalized_error =
            (reconstruction_error - self.baseline_error) / (self.max_error - self.baseline_error);

        normalized_error.clamp(0.0, 1.0)
    }
}

pub fn detect(model: &AutoEncoder, commit: &Commit) -> Result<f32> {
    let preprocessor = Preprocessor::new(PreprocessingConfig::default())?;
    let processed = preprocessor.preprocess_commit(commit);

    let tokenizer = create_tokenizer()?;
    let embeddings = preprocessor.create_embeddings(&[processed], &tokenizer)?;

    if embeddings.is_empty() || embeddings[0].is_empty() {
        return Ok(0.5);
    }

    let device = model.device();
    let input_dim = 768;

    let mut padded_embedding = embeddings[0].clone();
    padded_embedding.resize(input_dim, 0.0);

    let input_tensor = Tensor::from_vec(padded_embedding.clone(), &[1, input_dim], device)?
        .to_dtype(DType::F32)?;

    let latent = model.encode(&input_tensor)?;
    let reconstructed = model.decode(&latent)?;

    let input_flat = Tensor::from_vec(padded_embedding, input_dim, device)?;
    let reconstructed_flat = reconstructed.flatten_all()?;

    let diff = (&input_flat - &reconstructed_flat)?;
    let squared_diff = diff.sqr()?;
    let mse = squared_diff.mean_all()?;

    let reconstruction_error = mse.to_scalar::<f32>()?;

    let detector = VulnerabilityDetector::default();
    Ok(detector.calculate_score(reconstruction_error))
}

pub fn detect_batch(
    model: &AutoEncoder,
    commits: &[Commit],
    detector: &VulnerabilityDetector,
) -> Result<Vec<(String, f32)>> {
    let preprocessor = Preprocessor::new(PreprocessingConfig::default())?;
    let tokenizer = create_tokenizer()?;

    let mut results = Vec::new();

    for commit in commits {
        let processed = preprocessor.preprocess_commit(commit);
        let embeddings = preprocessor.create_embeddings(&[processed], &tokenizer)?;

        if embeddings.is_empty() || embeddings[0].is_empty() {
            results.push((commit.id.clone(), 0.5));
            continue;
        }

        let device = model.device();
        let input_dim = 768;

        let mut padded_embedding = embeddings[0].clone();
        padded_embedding.resize(input_dim, 0.0);

        let input_tensor = Tensor::from_vec(padded_embedding.clone(), &[1, input_dim], device)?
            .to_dtype(DType::F32)?;

        let latent = model.encode(&input_tensor)?;
        let reconstructed = model.decode(&latent)?;

        let input_flat = Tensor::from_vec(padded_embedding, input_dim, device)?;
        let reconstructed_flat = reconstructed.flatten_all()?;

        let diff = (&input_flat - &reconstructed_flat)?;
        let squared_diff = diff.sqr()?;
        let mse = squared_diff.mean_all()?;

        let reconstruction_error = mse.to_scalar::<f32>()?;
        let score = detector.calculate_score(reconstruction_error);

        results.push((commit.id.clone(), score));
    }

    Ok(results)
}

fn create_tokenizer() -> Result<tokenizers::Tokenizer> {
    // For now, use a simple character-level tokenizer
    // In production, you would load a pre-trained tokenizer
    use tokenizers::{models::bpe::BPE, Tokenizer};

    let tokenizer = Tokenizer::new(BPE::default());
    Ok(tokenizer)
}
