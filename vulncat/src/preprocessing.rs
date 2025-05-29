use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;
use crate::data::Commit;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreprocessingConfig {
    pub max_diff_length: usize,
    pub max_message_length: usize,
    pub remove_comments: bool,
    pub normalize_whitespace: bool,
    pub extract_code_features: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            max_diff_length: 5000,
            max_message_length: 500,
            remove_comments: true,
            normalize_whitespace: true,
            extract_code_features: true,
        }
    }
}

pub struct Preprocessor {
    config: PreprocessingConfig,
    vulnerability_patterns: Vec<Regex>,
}

impl Preprocessor {
    pub fn new(config: PreprocessingConfig) -> Result<Self> {
        let vulnerability_patterns = vec![
            Regex::new(r"(?i)(buffer|overflow|memory|leak|injection|xss|sql)")?,
            Regex::new(r"(?i)(strcpy|strcat|gets|sprintf|scanf)")?,
            Regex::new(r"(?i)(eval|exec|system|shell)")?,
            Regex::new(r#"(?i)(password|secret|key|token).*=.*['"]"#)?,
            Regex::new(r"(?i)(TODO|FIXME|HACK|XXX)")?,
        ];
        
        Ok(Self {
            config,
            vulnerability_patterns,
        })
    }
    
    pub fn preprocess_commit(&self, commit: &Commit) -> ProcessedCommit {
        let message = self.clean_text(&commit.message, self.config.max_message_length);
        let diff = self.clean_diff(&commit.diff, self.config.max_diff_length);
        
        let features = if self.config.extract_code_features {
            self.extract_features(&commit.message, &commit.diff)
        } else {
            CommitFeatures::default()
        };
        
        ProcessedCommit {
            id: commit.id.clone(),
            message,
            diff,
            features,
            label: commit.label,
        }
    }
    
    fn clean_text(&self, text: &str, max_length: usize) -> String {
        let mut cleaned = text.trim().to_string();
        
        if self.config.normalize_whitespace {
            cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        
        if cleaned.len() > max_length {
            cleaned.truncate(max_length);
        }
        
        cleaned
    }
    
    fn clean_diff(&self, diff: &str, max_length: usize) -> String {
        let lines: Vec<String> = diff
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                if self.config.remove_comments {
                    self.remove_comments(line)
                } else {
                    line.to_string()
                }
            })
            .collect();
        
        let mut result = lines.join(" \n");
        if result.len() > max_length {
            result.truncate(max_length);
        }
        
        result
    }
    
    fn remove_comments(&self, line: &str) -> String {
        let line = Regex::new(r"//.*$").unwrap().replace(line, "");
        let line = Regex::new(r"/\*.*?\*/").unwrap().replace_all(&line, "");
        line.to_string()
    }
    
    fn extract_features(&self, message: &str, diff: &str) -> CommitFeatures {
        let combined = format!("{} {}", message, diff);
        
        let vulnerability_keywords = self.vulnerability_patterns
            .iter()
            .map(|pattern| pattern.find_iter(&combined).count())
            .sum::<usize>();
        
        let added_lines = diff.lines().filter(|l| l.starts_with('+')).count();
        let removed_lines = diff.lines().filter(|l| l.starts_with('-')).count();
        let modified_files = diff.lines()
            .filter(|l| l.starts_with("+++") || l.starts_with("---"))
            .count() / 2;
        
        let code_complexity = self.estimate_complexity(diff);
        
        CommitFeatures {
            vulnerability_keywords: vulnerability_keywords as f32,
            added_lines: added_lines as f32,
            removed_lines: removed_lines as f32,
            modified_files: modified_files as f32,
            code_complexity,
            diff_size: diff.len() as f32,
            message_length: message.len() as f32,
        }
    }
    
    fn estimate_complexity(&self, diff: &str) -> f32 {
        let complexity_indicators = vec![
            r"\bif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b",
            r"\btry\b", r"\bcatch\b", r"\bswitch\b", r"\bcase\b",
        ];
        
        let mut complexity = 0.0;
        for indicator in complexity_indicators {
            if let Ok(regex) = Regex::new(indicator) {
                complexity += regex.find_iter(diff).count() as f32;
            }
        }
        
        complexity
    }
    
    pub fn create_embeddings(&self, commits: &[ProcessedCommit], tokenizer: &tokenizers::Tokenizer) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        
        for commit in commits {
            let text = format!("{}\n{}", commit.message, commit.diff);
            let encoding = tokenizer.encode(text, false)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
            let ids = encoding.get_ids();
            
            let mut embedding: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
            
            embedding.extend_from_slice(&[
                commit.features.vulnerability_keywords,
                commit.features.added_lines,
                commit.features.removed_lines,
                commit.features.modified_files,
                commit.features.code_complexity,
                commit.features.diff_size / 1000.0,
                commit.features.message_length / 100.0,
            ]);
            
            embeddings.push(embedding);
        }
        
        Ok(embeddings)
    }
}

#[derive(Debug, Clone)]
pub struct ProcessedCommit {
    pub id: String,
    pub message: String,
    pub diff: String,
    pub features: CommitFeatures,
    pub label: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub struct CommitFeatures {
    pub vulnerability_keywords: f32,
    pub added_lines: f32,
    pub removed_lines: f32,
    pub modified_files: f32,
    pub code_complexity: f32,
    pub diff_size: f32,
    pub message_length: f32,
}