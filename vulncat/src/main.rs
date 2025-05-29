use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use vulncat::{Config, VulnCat, data::CommitDataset, evaluation::Evaluator};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(short, long, value_name = "FILE")]
        dataset: PathBuf,
        
        #[arg(short, long, value_name = "FILE")]
        config: Option<PathBuf>,
        
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,
    },
    
    Detect {
        #[arg(short, long, value_name = "FILE")]
        commit: PathBuf,
        
        #[arg(short, long, value_name = "FILE")]
        model: PathBuf,
        
        #[arg(short, long)]
        threshold: Option<f32>,
    },
    
    Batch {
        #[arg(short, long, value_name = "FILE")]
        input: PathBuf,
        
        #[arg(short, long, value_name = "FILE")]
        model: PathBuf,
        
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
    
    Evaluate {
        #[arg(short, long, value_name = "FILE")]
        dataset: PathBuf,
        
        #[arg(short, long, value_name = "FILE")]
        model: Option<PathBuf>,
        
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Train { dataset, config, output } => {
            let config = if let Some(config_path) = config {
                let config_str = std::fs::read_to_string(config_path)?;
                serde_json::from_str(&config_str)?
            } else {
                Config::default()
            };
            
            let dataset = CommitDataset::from_json_file(dataset, config.data.clone())?;
            let mut vulncat = VulnCat::new(config)?;
            
            println!("Starting training with {} commits...", dataset.len());
            vulncat.train(&dataset)?;
            
            if let Some(output_dir) = output {
                std::fs::create_dir_all(&output_dir)?;
                println!("Model saved to {:?}", output_dir);
            }
            
            Ok(())
        }
        
        Commands::Detect { commit, model, threshold } => {
            let commit_str = std::fs::read_to_string(commit)?;
            let commit: vulncat::data::Commit = serde_json::from_str(&commit_str)?;
            
            let config = Config::default();
            let vulncat = VulnCat::new(config)?;
            
            let score = vulncat.detect(&commit)?;
            let threshold = threshold.unwrap_or(0.5);
            
            println!("Reconstruction error: {:.4}", score);
            println!("Vulnerability detected: {}", score > threshold);
            
            Ok(())
        }
        
        Commands::Batch { input, model, output } => {
            let config = Config::default();
            let vulncat = VulnCat::new(config)?;
            
            let commits_str = std::fs::read_to_string(input)?;
            let commits: Vec<vulncat::data::Commit> = serde_json::from_str(&commits_str)?;
            
            let mut results = Vec::new();
            
            for commit in &commits {
                let score = vulncat.detect(commit)?;
                results.push(serde_json::json!({
                    "commit_id": commit.id,
                    "score": score,
                    "is_vulnerable": score > 0.5
                }));
            }
            
            if let Some(output_path) = output {
                let results_json = serde_json::to_string_pretty(&results)?;
                std::fs::write(output_path, results_json)?;
                println!("Results saved to file");
            } else {
                println!("{}", serde_json::to_string_pretty(&results)?);
            }
            
            Ok(())
        }
        
        Commands::Evaluate { dataset, model, output } => {
            let dataset = CommitDataset::from_json_file(dataset, vulncat::data::DataConfig::default())?;
            println!("Loaded {} commits for evaluation", dataset.len());
            
            let config = Config::default();
            let model = if let Some(_model_path) = model {
                // TODO: Load model from checkpoint
                vulncat::model::AutoEncoder::new(&config.model)?
            } else {
                // Create new model for testing
                vulncat::model::AutoEncoder::new(&config.model)?
            };
            
            let evaluator = Evaluator::new(model);
            let report = evaluator.evaluate(&dataset)?;
            
            report.print_full_report();
            
            if let Some(output_path) = output {
                report.save_to_file(output_path.to_str().unwrap())?;
            }
            
            Ok(())
        }
    }
}