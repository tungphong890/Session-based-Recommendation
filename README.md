# Session-Based Recommendation System

A sophisticated machine learning system that generates personalized product recommendations based on user session behavior and historical patterns. This project implements state-of-the-art session-based recommendation algorithms optimized for e-commerce, streaming platforms, and content discovery applications.

## Overview

Session-based recommendation systems predict what users will interact with next based on their current session history, without requiring explicit user profiles. This approach is particularly valuable for:
- E-commerce platforms (Amazon, eBay)
- Streaming services (Netflix, YouTube)
- Social media platforms
- Ad recommendation systems
- Anonymous user scenarios

### Key Features
- **Advanced Algorithms**: RNN, GRU, and attention-based models
- **Real-time Predictions**: Sub-second recommendation latency
- **Session Encoding**: Captures sequential user behavior patterns
- **Personalized Rankings**: Context-aware recommendation ranking
- **Scalable Architecture**: Handles large-scale session data
- **Interpretability**: Explainable recommendation reasons
- **Performance Tracking**: Built-in metrics and monitoring

##  Architecture & Project Structure

```
Session-based-Recommendation/
├── README.md                       # This file
├── code/                          # Main implementation directory
│   ├── data_loader.py             # Session data loading and preprocessing
│   ├── models.py                  # Recommendation model implementations
│   │   ├── SessionRNN             # RNN-based session encoder
│   │   ├── SessionGRU             # GRU-based session encoder
│   │   └── AttentionModel         # Attention mechanism
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # Model evaluation metrics
│   ├── recommend.py               # Inference and recommendation
│   └── utils.py                   # Utility functions
│
└── Performance/                   # Results and benchmarks
    ├── metrics.json               # Performance metrics
    ├── plots/                     # Visualization plots
    │   ├── recall_comparison.png
    │   ├── ndcg_scores.png
    │   └── training_curves.png
    └── results.txt                # Detailed results
```

### System Architecture

```
User Session Input
       ↓
Data Preprocessing
       ↓
Session Encoding (RNN/GRU)
       ↓
Attention Mechanism
       ↓
Item Embedding Layer
       ↓
Scoring & Ranking
       ↓
Top-K Recommendations
```

##  Installation Guide

### Prerequisites
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended for training)
- **Storage**: 2GB for code and datasets
- **Optional**: GPU (NVIDIA) for faster training

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/tungphong890/Session-based-Recommendation.git
cd Session-based-Recommendation
```

#### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
# Core dependencies
pip install numpy pandas scikit-learn
pip install torch torchvision torchaudio
pip install tensorflow  # Alternative to PyTorch
pip install matplotlib seaborn
pip install jupyter notebook  # For analysis
```

#### 4. Prepare Dataset
```bash
# Download your e-commerce session data
# Expected format: CSV with columns [user_id, session_id, item_id, timestamp]
mkdir -p data
cp your_session_data.csv data/sessions.csv
```

#### 5. Verify Installation
```bash
python -c "import torch; import tensorflow as tf; print('Ready!')"
```

##  Usage Guide

### Basic Workflow

#### 1. Load and Prepare Data
```python
from code.data_loader import SessionDataLoader

# Initialize data loader
loader = SessionDataLoader(
    data_path='data/sessions.csv',
    min_session_length=2,
    min_item_freq=5
)

# Split into train/validation/test
train_data, val_data, test_data = loader.split(
    ratios=[0.7, 0.1, 0.2]
)
```

#### 2. Train Recommendation Model
```python
from code.models import SessionGRU
from code.train import Trainer

# Initialize model
model = SessionGRU(
    vocab_size=loader.num_items,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.3,
    attention=True
)

# Train model
trainer = Trainer(
    model=model,
    learning_rate=0.001,
    batch_size=32,
    epochs=50,
    device='cuda'  # or 'cpu'
)

history = trainer.train(
    train_data=train_data,
    val_data=val_data
)
```

#### 3. Generate Recommendations
```python
from code.recommend import Recommender

# Initialize recommender
recommender = Recommender(
    model=model,
    item_encoder=loader.item_encoder,
    top_k=10
)

# Get recommendations for a session
session = [item_1, item_2, item_3]  # User's current session
recommendations = recommender.recommend(
    session=session,
    n_recommendations=5,
    diversity=0.7
)

for rank, (item_id, score) in enumerate(recommendations, 1):
    print(f"{rank}. Item {item_id} (Score: {score:.4f})")
```

#### 4. Evaluate Model
```python
from code.evaluate import Evaluator

evaluator = Evaluator(recommender)
metrics = evaluator.evaluate(test_data)

print(f"Recall@10: {metrics['recall@10']:.4f}")
print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
print(f"MRR: {metrics['mrr']:.4f}")
print(f"Coverage: {metrics['coverage']:.4f}")
```

### Advanced Configuration

#### Model Parameters
```python
model = SessionGRU(
    vocab_size=10000,              # Number of unique items
    embedding_dim=128,             # Item embedding dimension
    hidden_dim=256,                # RNN hidden dimension
    num_layers=2,                  # Stacked RNN layers
    dropout=0.3,                   # Dropout rate
    attention=True,                # Enable attention
    attention_dim=64,              # Attention dimension
    bidirectional=False            # Unidirectional RNN
)
```

#### Training Configuration
```python
trainer = Trainer(
    model=model,
    learning_rate=0.001,
    batch_size=32,
    epochs=50,
    patience=10,                   # Early stopping patience
    device='cuda',                 # GPU device
    loss_function='cross_entropy',
    optimizer='adam',
    weight_decay=1e-5,
    gradient_clip=1.0
)
```

##  Model Types & Algorithms

### 1. Session RNN
- Uses vanilla RNN for sequential modeling
- Best for: Simple, small-scale sessions
- Trade-off: Lower accuracy but faster training
- Complexity: O(n) where n = session length

### 2. Session GRU (Recommended)
- Gated Recurrent Unit with fewer parameters than LSTM
- Best for: Balanced speed and accuracy
- Advantages: Faster training than LSTM
- Complexity: O(n * hidden_dim²)

### 3. Attention-Based Model
- Incorporates multi-head attention mechanism
- Best for: Long sessions with variable importance items
- Advantages: Interpretable attention weights
- Complexity: O(n² * hidden_dim)

##  Performance Metrics

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Recall@K** | Items in top-K / Total relevant items | % of relevant items recalled |
| **NDCG@K** | Normalized DCG / Ideal DCG | Ranking quality 0-1 |
| **MRR** | 1 / Rank of first relevant item | Mean reciprocal rank |
| **Coverage** | Unique items recommended / Total items | Item diversity coverage |
| **Diversity** | 1 - avg(similarity) | Recommendation variety |

### Expected Performance (Benchmark)
```
Dataset: Yoochoose (E-commerce sessions)
Model: Session GRU with Attention

Metric          | Score
----------------|--------
Recall@20       | 0.72
NDCG@20         | 0.58
MRR             | 0.34
Coverage        | 0.85
```

##  Best Practices

### Data Preparation
1. **Session Definition**
   - Clear session boundaries (timeout-based or explicit)
   - Minimum session length ≥ 2 items
   - Remove noise and bot sessions

2. **Temporal Considerations**
   - Include timestamp information
   - Handle seasonal patterns
   - Account for concept drift

3. **Item Encoding**
   - Handle cold-start items carefully
   - Use item metadata when available
   - Balance popularity bias

### Model Training
1. **Hyperparameter Tuning**
   ```python
   # Grid search example
   params = {
       'hidden_dim': [64, 128, 256],
       'embedding_dim': [64, 128],
       'dropout': [0.2, 0.3, 0.5],
       'learning_rate': [0.0001, 0.001, 0.01]
   }
   ```

2. **Avoiding Overfitting**
   - Use dropout (0.2-0.5)
   - Early stopping with patience
   - Regular validation monitoring
   - Cross-validation for final metrics

3. **Production Deployment**
   - Save best model checkpoint
   - Version control models
   - A/B test recommendations
   - Monitor recommendation diversity

### Inference Optimization
```python
# Batch recommendation inference
sessions_batch = [session_1, session_2, ...]
recommendations = recommender.recommend_batch(
    sessions=sessions_batch,
    n_recommendations=5,
    batch_size=64
)
```

##  Configuration & Customization

### Dataset Configuration
```python
config = {
    'data_path': 'data/sessions.csv',
    'min_session_length': 2,
    'min_item_freq': 5,
    'session_timeout': 1800,  # 30 minutes
    'date_format': '%Y-%m-%d %H:%M:%S'
}
```

### Model Configuration
```python
model_config = {
    'vocab_size': 10000,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'use_attention': True,
    'attention_heads': 4
}
```

##  Advanced Topics

### Cold-Start Recommendations
```python
# For new items without history
recommendations = recommender.recommend_cold_start(
    num_recommendations=5,
    strategy='popularity',  # or 'random', 'content-based'
    metadata=item_metadata
)
```

### Cross-Session Context
```python
# Include user profile information
recommendations = recommender.recommend_with_context(
    session=current_session,
    user_profile=user_history,
    time_context=current_time,
    location=user_location
)
```

### Diversity Control
```python
# Generate diverse recommendations
recommendations = recommender.recommend_diverse(
    session=session,
    n_recommendations=10,
    diversity_weight=0.7  # 0=relevance, 1=diversity
)
```

##  Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Out of memory | Large batch size | Reduce batch size, use gradient accumulation |
| Low recall scores | Poor data quality | Clean sessions, filter noise |
| Slow inference | Model too large | Quantize model, use smaller hidden_dim |
| Overfitting | Too few samples | Increase dropout, more regularization |
| GPU not detected | CUDA not installed | Install GPU PyTorch version |

##  Visualization & Analysis

```python
# Training curves
trainer.plot_training_history()

# Recommendation diversity
evaluator.plot_diversity_distribution()

# Attention weights visualization
recommender.visualize_attention(session)

# Performance comparison
evaluator.compare_models([model1, model2, model3])
```

##  Resources & References

- [Session-based Recommendations with RNNs](https://arxiv.org/abs/1506.01084)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch RNN Documentation](https://pytorch.org/docs/stable/nn.html#rnn-layers)
- [Recommendation Systems Handbook](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1)

##  Performance Directory

Check `/Performance` folder for:
- Detailed metrics and benchmarks
- Training curves and visualizations
- Comparison with baseline models
- Performance on different datasets

##  License

MIT License - see LICENSE file for details

##  Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with improvements

##  Acknowledgments

- Original RNN-based recommendation paper authors
- PyTorch and TensorFlow communities
- Dataset providers (Yoochoose, etc.)

---

**Last Updated**: March 2026  
**Version**: 1.0.0
