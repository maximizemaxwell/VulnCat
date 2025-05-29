use vulncat::{Config, VulnCat, data::{Commit, CommitDataset}};

#[test]
fn test_model_creation() {
    let config = Config::default();
    let vulncat = VulnCat::new(config);
    assert!(vulncat.is_ok());
}

#[test]
fn test_commit_dataset() {
    let commits = vec![
        Commit {
            id: "test123".to_string(),
            message: "Fix buffer overflow".to_string(),
            diff: "+ if (index < size) { buffer[index] = value; }".to_string(),
            author: "test@example.com".to_string(),
            timestamp: "2024-01-01".to_string(),
            label: Some(true),
        }
    ];
    
    let dataset = CommitDataset::new(commits, vulncat::data::DataConfig::default());
    assert_eq!(dataset.len(), 1);
    assert_eq!(dataset.num_batches(), 1);
}

#[test]
fn test_detection() {
    let config = Config::default();
    let vulncat = VulnCat::new(config).unwrap();
    
    let commit = Commit {
        id: "test456".to_string(),
        message: "Update dependencies".to_string(),
        diff: "- version = \"1.0\"\n+ version = \"1.1\"".to_string(),
        author: "dev@example.com".to_string(),
        timestamp: "2024-01-02".to_string(),
        label: None,
    };
    
    let result = vulncat.detect(&commit);
    assert!(result.is_ok());
}