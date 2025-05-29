use anyhow::Result;
use std::collections::HashMap;

use crate::{
    data::{Commit, CommitDataset},
    inference::{detect_batch, VulnerabilityDetector},
    model::AutoEncoder,
};

#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub auc_roc: f32,
    pub true_positives: usize,
    pub true_negatives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub threshold: f32,
}

impl EvaluationMetrics {
    pub fn new(predictions: &[(String, f32, Option<bool>)], threshold: f32) -> Self {
        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut false_negatives = 0;

        for (_, score, label) in predictions {
            if let Some(is_vulnerable) = label {
                let predicted_vulnerable = *score > threshold;

                match (predicted_vulnerable, *is_vulnerable) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => false_negatives += 1,
                    (false, false) => tn += 1,
                }
            }
        }

        let accuracy = if tp + tn + fp + false_negatives > 0 {
            (tp + tn) as f32 / (tp + tn + fp + false_negatives) as f32
        } else {
            0.0
        };

        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };

        let recall = if tp + false_negatives > 0 {
            tp as f32 / (tp + false_negatives) as f32
        } else {
            0.0
        };

        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let auc_roc = calculate_auc_roc(predictions);

        Self {
            accuracy,
            precision,
            recall,
            f1_score,
            auc_roc,
            true_positives: tp,
            true_negatives: tn,
            false_positives: fp,
            false_negatives,
            threshold,
        }
    }

    pub fn print_report(&self) {
        println!("\n=== Evaluation Metrics ===");
        println!("Threshold: {:.2}", self.threshold);
        println!(
            "Accuracy: {:.4} ({:.2}%)",
            self.accuracy,
            self.accuracy * 100.0
        );
        println!("Precision: {:.4}", self.precision);
        println!("Recall: {:.4}", self.recall);
        println!("F1-Score: {:.4}", self.f1_score);
        println!("AUC-ROC: {:.4}", self.auc_roc);
        println!("\nConfusion Matrix:");
        println!("  True Positives: {}", self.true_positives);
        println!("  True Negatives: {}", self.true_negatives);
        println!("  False Positives: {}", self.false_positives);
        println!("  False Negatives: {}", self.false_negatives);
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "true_negatives": self.true_negatives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
            }
        })
    }
}

pub struct Evaluator {
    model: AutoEncoder,
    detector: VulnerabilityDetector,
}

impl Evaluator {
    pub fn new(model: AutoEncoder) -> Self {
        Self {
            model,
            detector: VulnerabilityDetector::default(),
        }
    }

    pub fn evaluate(&self, dataset: &CommitDataset) -> Result<EvaluationReport> {
        let predictions = self.get_predictions(dataset)?;

        let thresholds = vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let mut best_threshold = 0.5;
        let mut best_f1 = 0.0;
        let mut threshold_metrics = HashMap::new();

        for &threshold in &thresholds {
            let metrics = EvaluationMetrics::new(&predictions, threshold);
            if metrics.f1_score > best_f1 {
                best_f1 = metrics.f1_score;
                best_threshold = threshold;
            }
            threshold_metrics.insert(threshold.to_string(), metrics);
        }

        let best_metrics = EvaluationMetrics::new(&predictions, best_threshold);

        Ok(EvaluationReport {
            best_metrics,
            best_threshold,
            threshold_metrics,
            predictions,
        })
    }

    fn get_predictions(&self, dataset: &CommitDataset) -> Result<Vec<(String, f32, Option<bool>)>> {
        let commits: Vec<Commit> = dataset.iter().cloned().collect();
        let scores = detect_batch(&self.model, &commits, &self.detector)?;

        let mut predictions = Vec::new();
        for (commit, (id, score)) in commits.iter().zip(scores.iter()) {
            predictions.push((id.clone(), *score, commit.label));
        }

        Ok(predictions)
    }
}

#[derive(Debug)]
pub struct EvaluationReport {
    pub best_metrics: EvaluationMetrics,
    pub best_threshold: f32,
    pub threshold_metrics: HashMap<String, EvaluationMetrics>,
    pub predictions: Vec<(String, f32, Option<bool>)>,
}

impl EvaluationReport {
    pub fn print_full_report(&self) {
        println!("\n===== EVALUATION REPORT =====");
        println!("Best threshold: {:.2}", self.best_threshold);
        self.best_metrics.print_report();

        println!("\n=== Metrics by Threshold ===");
        let mut thresholds: Vec<_> = self.threshold_metrics.keys().collect();
        thresholds.sort();

        for threshold in thresholds {
            let metrics = &self.threshold_metrics[threshold];
            println!(
                "Threshold {}: Acc={:.3}, Prec={:.3}, Rec={:.3}, F1={:.3}",
                threshold, metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score
            );
        }

        println!("\n=== Sample Predictions ===");
        let mut sorted_predictions = self.predictions.clone();
        sorted_predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Top 5 highest risk commits:");
        for (id, score, label) in sorted_predictions.iter().take(5) {
            let label_str = match label {
                Some(true) => "vulnerable",
                Some(false) => "safe",
                None => "unknown",
            };
            println!("  {} - Score: {:.4} ({})", id, score, label_str);
        }

        println!("\nTop 5 lowest risk commits:");
        for (id, score, label) in sorted_predictions.iter().rev().take(5) {
            let label_str = match label {
                Some(true) => "vulnerable",
                Some(false) => "safe",
                None => "unknown",
            };
            println!("  {} - Score: {:.4} ({})", id, score, label_str);
        }
    }

    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let report_json = serde_json::json!({
            "best_threshold": self.best_threshold,
            "best_metrics": self.best_metrics.to_json(),
            "threshold_analysis": self.threshold_metrics.iter()
                .map(|(k, v)| (k.clone(), v.to_json()))
                .collect::<HashMap<_, _>>(),
            "predictions": self.predictions.iter()
                .map(|(id, score, label)| {
                    serde_json::json!({
                        "commit_id": id,
                        "vulnerability_score": score,
                        "actual_label": label,
                    })
                })
                .collect::<Vec<_>>(),
        });

        let json_string = serde_json::to_string_pretty(&report_json)?;
        std::fs::write(path, json_string)?;
        println!("Evaluation report saved to: {}", path);

        Ok(())
    }
}

fn calculate_auc_roc(predictions: &[(String, f32, Option<bool>)]) -> f32 {
    let mut labeled_predictions: Vec<(f32, bool)> = predictions
        .iter()
        .filter_map(|(_, score, label)| label.map(|l| (*score, l)))
        .collect();

    if labeled_predictions.is_empty() {
        return 0.5;
    }

    labeled_predictions.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let total_positives = labeled_predictions.iter().filter(|(_, l)| *l).count();
    let total_negatives = labeled_predictions.len() - total_positives;

    if total_positives == 0 || total_negatives == 0 {
        return 0.5;
    }

    let mut auc = 0.0;
    let mut tp = 0;
    let mut fp = 0;
    let mut prev_score = 1.0;
    let mut prev_tpr = 0.0;
    let mut prev_fpr = 0.0;

    for (score, is_positive) in &labeled_predictions {
        if *score != prev_score {
            let tpr = tp as f32 / total_positives as f32;
            let fpr = fp as f32 / total_negatives as f32;

            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

            prev_tpr = tpr;
            prev_fpr = fpr;
            prev_score = *score;
        }

        if *is_positive {
            tp += 1;
        } else {
            fp += 1;
        }
    }

    let tpr = tp as f32 / total_positives as f32;
    let fpr = fp as f32 / total_negatives as f32;
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

    auc
}
