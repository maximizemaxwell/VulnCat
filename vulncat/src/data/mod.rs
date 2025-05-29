use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub batch_size: usize,
    pub max_sequence_length: usize,
    pub shuffle: bool,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_sequence_length: 512,
            shuffle: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    pub id: String,
    pub message: String,
    pub diff: String,
    pub author: String,
    pub timestamp: String,
    pub label: Option<bool>,
}

pub struct CommitDataset {
    commits: Vec<Commit>,
    pub config: DataConfig,
}

impl CommitDataset {
    pub fn new(commits: Vec<Commit>, config: DataConfig) -> Self {
        Self { commits, config }
    }
    
    pub fn from_json_file<P: AsRef<Path>>(path: P, config: DataConfig) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let commits: Vec<Commit> = serde_json::from_str(&content)?;
        Ok(Self::new(commits, config))
    }
    
    pub fn len(&self) -> usize {
        self.commits.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.commits.is_empty()
    }
    
    pub fn get_batch(&self, idx: usize) -> Option<&[Commit]> {
        let start = idx * self.config.batch_size;
        let end = ((idx + 1) * self.config.batch_size).min(self.commits.len());
        
        if start >= self.commits.len() {
            None
        } else {
            Some(&self.commits[start..end])
        }
    }
    
    pub fn num_batches(&self) -> usize {
        (self.commits.len() + self.config.batch_size - 1) / self.config.batch_size
    }
    
    pub fn iter(&self) -> impl Iterator<Item = &Commit> {
        self.commits.iter()
    }
}

pub fn encode_commit(commit: &Commit, tokenizer: &tokenizers::Tokenizer) -> Result<Vec<f32>> {
    let text = format!("{}\n{}", commit.message, commit.diff);
    let encoding = tokenizer.encode(text, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let ids = encoding.get_ids();
    
    Ok(ids.iter().map(|&id| id as f32).collect())
}