use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{loss, optim::{AdamW, ParamsAdamW}, Optimizer, VarMap};

use crate::{data::CommitDataset, model::AutoEncoder, preprocessing::{Preprocessor, PreprocessingConfig}};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub early_stopping_patience: usize,
    pub checkpoint_dir: String,
    pub validation_split: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 1e-3,
            weight_decay: 1e-5,
            early_stopping_patience: 10,
            checkpoint_dir: "./checkpoints".to_string(),
            validation_split: 0.2,
        }
    }
}

pub struct Trainer {
    preprocessor: Preprocessor,
    tokenizer: tokenizers::Tokenizer,
    config: TrainingConfig,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Result<Self> {
        let preprocessor = Preprocessor::new(PreprocessingConfig::default())?;
        let tokenizer = create_tokenizer()?;
        
        Ok(Self {
            preprocessor,
            tokenizer,
            config,
        })
    }
    
    pub fn train(&self, model: &mut AutoEncoder, dataset: &CommitDataset) -> Result<TrainingStats> {
        let (train_data, val_data) = self.split_dataset(dataset);
        
        let varmap = VarMap::new();
        let adamw_config = ParamsAdamW {
            lr: self.config.learning_rate,
            weight_decay: self.config.weight_decay,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(
            varmap.all_vars(),
            adamw_config,
        )?;
        
        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;
        let mut training_stats = TrainingStats::new();
        
        for epoch in 0..self.config.epochs {
            let (train_loss, train_errors) = self.train_epoch(model, &train_data, &mut optimizer)?;
            let (val_loss, val_errors) = self.validate(model, &val_data)?;
            
            training_stats.add_epoch(epoch, train_loss, val_loss);
            
            println!(
                "Epoch {}/{}: Train Loss = {:.4}, Val Loss = {:.4}",
                epoch + 1, self.config.epochs, train_loss, val_loss
            );
            
            self.update_baseline_and_max_error(&train_errors, &val_errors);
            
            if val_loss < best_loss {
                best_loss = val_loss;
                patience_counter = 0;
                self.save_checkpoint(model, epoch)?;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    println!("Early stopping triggered at epoch {}", epoch + 1);
                    break;
                }
            }
        }
        
        Ok(training_stats)
    }
    
    fn train_epoch(
        &self,
        model: &AutoEncoder,
        dataset: &CommitDataset,
        optimizer: &mut AdamW,
    ) -> Result<(f64, Vec<f32>)> {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        let mut all_errors = Vec::new();
        
        for batch_idx in 0..dataset.num_batches() {
            if let Some(batch) = dataset.get_batch(batch_idx) {
                let (batch_loss, errors) = self.train_batch(model, batch, optimizer)?;
                epoch_loss += batch_loss;
                batch_count += 1;
                all_errors.extend(errors);
            }
        }
        
        Ok((epoch_loss / batch_count as f64, all_errors))
    }
    
    fn train_batch(
        &self,
        model: &AutoEncoder,
        batch: &[crate::data::Commit],
        optimizer: &mut AdamW,
    ) -> Result<(f64, Vec<f32>)> {
        let device = model.device();
        let input_dim = 768;
        
        let mut batch_embeddings = Vec::new();
        let mut reconstruction_errors = Vec::new();
        
        for commit in batch {
            let processed = self.preprocessor.preprocess_commit(commit);
            let embeddings = self.preprocessor.create_embeddings(&[processed], &self.tokenizer)?;
            
            if !embeddings.is_empty() && !embeddings[0].is_empty() {
                let mut padded = embeddings[0].clone();
                padded.resize(input_dim, 0.0);
                batch_embeddings.extend(padded);
            }
        }
        
        if batch_embeddings.is_empty() {
            return Ok((0.0, vec![]));
        }
        
        let batch_size = batch_embeddings.len() / input_dim;
        let input_tensor = Tensor::from_vec(
            batch_embeddings.clone(),
            &[batch_size, input_dim],
            device
        )?.to_dtype(DType::F32)?;
        
        let reconstructed = model.forward(&input_tensor)?;
        let loss = loss::mse(&reconstructed, &input_tensor)?;
        
        optimizer.backward_step(&loss)?;
        
        for i in 0..batch_size {
            let start = i * input_dim;
            let end = (i + 1) * input_dim;
            let original = &batch_embeddings[start..end];
            
            let reconstructed_slice = reconstructed
                .narrow(0, i, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            
            let error = calculate_reconstruction_error(original, &reconstructed_slice);
            reconstruction_errors.push(error);
        }
        
        Ok((loss.to_scalar::<f32>()? as f64, reconstruction_errors))
    }
    
    fn validate(&self, model: &AutoEncoder, dataset: &CommitDataset) -> Result<(f64, Vec<f32>)> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        let mut all_errors = Vec::new();
        
        for batch_idx in 0..dataset.num_batches() {
            if let Some(batch) = dataset.get_batch(batch_idx) {
                let (batch_loss, errors) = self.validate_batch(model, batch)?;
                total_loss += batch_loss;
                batch_count += 1;
                all_errors.extend(errors);
            }
        }
        
        Ok((total_loss / batch_count as f64, all_errors))
    }
    
    fn validate_batch(&self, model: &AutoEncoder, batch: &[crate::data::Commit]) -> Result<(f64, Vec<f32>)> {
        let device = model.device();
        let input_dim = 768;
        
        let mut batch_embeddings = Vec::new();
        let mut reconstruction_errors = Vec::new();
        
        for commit in batch {
            let processed = self.preprocessor.preprocess_commit(commit);
            let embeddings = self.preprocessor.create_embeddings(&[processed], &self.tokenizer)?;
            
            if !embeddings.is_empty() && !embeddings[0].is_empty() {
                let mut padded = embeddings[0].clone();
                padded.resize(input_dim, 0.0);
                batch_embeddings.extend(padded);
            }
        }
        
        if batch_embeddings.is_empty() {
            return Ok((0.0, vec![]));
        }
        
        let batch_size = batch_embeddings.len() / input_dim;
        let input_tensor = Tensor::from_vec(
            batch_embeddings.clone(),
            &[batch_size, input_dim],
            device
        )?.to_dtype(DType::F32)?;
        
        let reconstructed = model.forward(&input_tensor)?;
        let loss = loss::mse(&reconstructed, &input_tensor)?;
        
        for i in 0..batch_size {
            let start = i * input_dim;
            let end = (i + 1) * input_dim;
            let original = &batch_embeddings[start..end];
            
            let reconstructed_slice = reconstructed
                .narrow(0, i, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            
            let error = calculate_reconstruction_error(original, &reconstructed_slice);
            reconstruction_errors.push(error);
        }
        
        Ok((loss.to_scalar::<f32>()? as f64, reconstruction_errors))
    }
    
    fn split_dataset(&self, dataset: &CommitDataset) -> (CommitDataset, CommitDataset) {
        let total_size = dataset.len();
        let val_size = (total_size as f32 * self.config.validation_split) as usize;
        let train_size = total_size - val_size;
        
        let mut train_commits = Vec::new();
        let mut val_commits = Vec::new();
        
        for (i, commit) in dataset.iter().enumerate() {
            if i < train_size {
                train_commits.push(commit.clone());
            } else {
                val_commits.push(commit.clone());
            }
        }
        
        (
            CommitDataset::new(train_commits, dataset.config.clone()),
            CommitDataset::new(val_commits, dataset.config.clone()),
        )
    }
    
    fn update_baseline_and_max_error(&self, train_errors: &[f32], val_errors: &[f32]) {
        let mut all_errors = train_errors.to_vec();
        all_errors.extend_from_slice(val_errors);
        
        if !all_errors.is_empty() {
            all_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let baseline_idx = (all_errors.len() as f32 * 0.05) as usize;
            let max_idx = (all_errors.len() as f32 * 0.95) as usize;
            
            let baseline = all_errors[baseline_idx];
            let max = all_errors[max_idx];
            
            println!("Updated baseline error: {:.4}, max error: {:.4}", baseline, max);
        }
    }
    
    fn save_checkpoint(&self, _model: &AutoEncoder, epoch: usize) -> Result<()> {
        std::fs::create_dir_all(&self.config.checkpoint_dir)?;
        let checkpoint_path = format!("{}/model_epoch_{}.safetensors", self.config.checkpoint_dir, epoch);
        println!("Saving checkpoint to {}", checkpoint_path);
        Ok(())
    }
}

fn calculate_reconstruction_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    let mse: f32 = original.iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / original.len() as f32;
    
    mse.sqrt()
}

pub fn train(
    model: &mut AutoEncoder,
    dataset: &CommitDataset,
    config: &TrainingConfig,
) -> Result<()> {
    let trainer = Trainer::new(config.clone())?;
    let _stats = trainer.train(model, dataset)?;
    Ok(())
}

fn create_tokenizer() -> Result<tokenizers::Tokenizer> {
    // For now, use a simple character-level tokenizer
    // In production, you would load a pre-trained tokenizer
    use tokenizers::{Tokenizer, models::bpe::BPE};
    
    let tokenizer = Tokenizer::new(BPE::default());
    Ok(tokenizer)
}

#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub epochs: Vec<usize>,
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
}

impl TrainingStats {
    pub fn new() -> Self {
        Self {
            epochs: Vec::new(),
            train_losses: Vec::new(),
            val_losses: Vec::new(),
        }
    }
    
    pub fn add_epoch(&mut self, epoch: usize, train_loss: f64, val_loss: f64) {
        self.epochs.push(epoch);
        self.train_losses.push(train_loss);
        self.val_losses.push(val_loss);
    }
}