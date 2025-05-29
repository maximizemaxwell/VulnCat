# VulnCat 🐱‍💻

VulnCat is an autoencoder-based vulnerability detection system that analyzes git commits to identify potential security vulnerabilities. It uses deep learning with the Candle framework to learn patterns from known vulnerable commits and scores new commits on a scale of 0-1.

## Features

- **Autoencoder Architecture**: Learns to reconstruct normal commit patterns, with higher reconstruction errors indicating potential vulnerabilities
- **Comprehensive Preprocessing**: Extracts both textual and statistical features from commits
- **Flexible Scoring**: Vulnerability scores range from 0-1 for easy interpretation
- **Evaluation Suite**: Built-in metrics including accuracy, precision, recall, F1-score, and AUC-ROC
- **GPU Support**: Automatically uses CUDA when available for faster processing
- **CLI Interface**: Easy-to-use command-line tools for training, detection, and evaluation

## Installation

### Prerequisites

- Rust 1.75 or higher
- CUDA toolkit (optional, for GPU acceleration)
- Git

### Build from Source

```bash
git clone https://github.com/yourusername/vulncat.git
cd vulncat/vulncat
cargo build --release
```

## Usage

### Command Overview

```bash
vulncat --help
```

Available commands:
- `train` - Train the autoencoder on a vulnerability dataset
- `detect` - Analyze a single commit for vulnerabilities
- `batch` - Process multiple commits at once
- `evaluate` - Evaluate model performance on labeled data

### Training a Model

Train the autoencoder on your vulnerability dataset:

```bash
vulncat train -d dataset/vulnerabilities.json -o checkpoints/
```

Options:
- `-d, --dataset <FILE>` - Path to training dataset (required)
- `-c, --config <FILE>` - Custom configuration file (optional)
- `-o, --output <DIR>` - Directory to save model checkpoints (optional)

### Detecting Vulnerabilities

#### Single Commit Analysis

```bash
vulncat detect -c commit.json -m model.bin --threshold 0.7
```

Options:
- `-c, --commit <FILE>` - Path to commit file (required)
- `-m, --model <FILE>` - Path to trained model (required)
- `--threshold <FLOAT>` - Detection threshold (default: 0.5)

#### Batch Processing

```bash
vulncat batch -i commits.json -m model.bin -o results.json
```

Options:
- `-i, --input <FILE>` - Input file with multiple commits (required)
- `-m, --model <FILE>` - Path to trained model (required)
- `-o, --output <FILE>` - Output file for results (optional)

### Model Evaluation

Evaluate model performance on labeled test data:

```bash
vulncat evaluate -d test_dataset.json -m model.bin -o evaluation_report.json
```

Options:
- `-d, --dataset <FILE>` - Path to labeled dataset (required)
- `-m, --model <FILE>` - Path to trained model (optional, creates new if not provided)
- `-o, --output <FILE>` - Save evaluation report (optional)

## Data Format

### Commit Format

Commits should be in JSON format with the following structure:

```json
{
  "id": "commit_hash",
  "message": "Fix buffer overflow in parse function",
  "diff": "- strcpy(buffer, input);\n+ strncpy(buffer, input, sizeof(buffer));",
  "author": "developer@example.com",
  "timestamp": "2024-01-15T10:30:00Z",
  "label": true  // optional, for training/evaluation
}
```

### Dataset Format

Multiple commits in a JSON array:

```json
[
  {
    "id": "abc123",
    "message": "...",
    "diff": "...",
    "author": "...",
    "timestamp": "...",
    "label": true
  },
  // more commits...
]
```

## Configuration

Create a custom configuration file to tune the model:

```json
{
  "model": {
    "input_dim": 768,
    "hidden_dim": 256,
    "latent_dim": 64,
    "dropout": 0.1
  },
  "training": {
    "epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 0.00001,
    "early_stopping_patience": 10,
    "validation_split": 0.2
  },
  "data": {
    "batch_size": 32,
    "max_sequence_length": 512,
    "shuffle": true
  }
}
```

## Understanding Vulnerability Scores

- **0.0 - 0.3**: Low risk - Commit patterns are similar to normal, safe commits
- **0.3 - 0.7**: Medium risk - Some unusual patterns detected, manual review recommended
- **0.7 - 1.0**: High risk - Strong indicators of potential vulnerabilities

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Vulnerability Scan
on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install VulnCat
        run: |
          cargo install --path vulncat/vulncat
      
      - name: Download Model
        run: |
          wget https://example.com/vulncat-model.bin
      
      - name: Analyze Commits
        run: |
          git log --format="%H" -n 10 | while read commit; do
            git show $commit > commit.json
            vulncat detect -c commit.json -m vulncat-model.bin
          done
```

## Performance Tips

1. **GPU Acceleration**: The system automatically uses CUDA if available. Ensure CUDA drivers are installed for best performance.

2. **Batch Processing**: Use the `batch` command for analyzing multiple commits - it's more efficient than individual detection.

3. **Model Size**: Adjust `hidden_dim` and `latent_dim` based on your dataset size. Larger models may be more accurate but slower.

4. **Preprocessing**: The preprocessing step can be computationally intensive. Consider caching preprocessed data for large datasets.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` in configuration
2. **Slow Training**: Enable GPU support or reduce model dimensions
3. **Poor Detection**: Ensure training dataset has balanced vulnerable/safe examples
4. **Tokenization Errors**: Check that commit text is properly UTF-8 encoded

### Debug Mode

Run with Rust's debug output for detailed information:

```bash
RUST_LOG=debug vulncat train -d dataset.json
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GPL-3.0 license - see the LICENSE file for details.

## Acknowledgments

- Built with [Candle](https://github.com/huggingface/candle) for efficient deep learning in Rust
- Inspired by research in automated vulnerability detection
- Thanks to the DiverseVul dataset contributors
