# Deep Learning for Sequential Data - RNNs

## Overview
This repository implements **Recurrent Neural Networks (RNNs), GRUs, and LSTMs** for sequence classification, focusing on text classification tasks. The project involves fundamental RNN implementation and deep learning techniques for sequential data, including manual RNN cell computation and backpropagation.

## Dataset
The dataset consists of labeled questions categorized into six classes:

- **Abbreviation (ABBR)**
- **Entity (ENTY)**
- **Description (DESC)**
- **Human (HUM)**
- **Location (LOC)**
- **Numeric (NUM)**

### **Data Split**
- **80%** training set
- **10%** validation set
- **10%** test set

## Model Architecture
This repository implements different RNN-based architectures:
- **Simple RNN** (manually implemented multi-timestep recurrent network)
- **GRU (Gated Recurrent Units)**
- **LSTM (Long Short-Term Memory)**

### **Key Implementations**
1. **Manual RNN Cell Computation:**
   - Declaring **U, W, V** matrices along with biases.
   - Computing hidden states using **tanh activation function**.
   - Calculating logits from hidden states.
   - Using **Cross-Entropy Loss** and performing backpropagation.
   
2. **Deep Learning with PyTorch:**
   - Using PyTorch to implement RNNs, GRUs, and LSTMs.
   - Training models using **Adam Optimizer**.
   - Evaluating model performance on **accuracy and loss metrics**.

### **Results**
| Model          | Train Accuracy | Validation Accuracy |
|---------------|---------------|---------------------|
| Simple RNN    | ~91.88%       | ~91.92%             |
| GRU           | ~95.30%       | ~94.80%             |
| LSTM          | ~98.25%       | ~96.97%             |

## Installation
To run this project, install the required dependencies:
```bash
pip install torch transformers datasets numpy pandas scikit-learn matplotlib
```

## How to Run
### 1. Clone the Repository
```bash
git clone https://github.com/mohibkhann/RNNs-Sequence-Classification.git
cd RNNs-Sequence-Classification
```

### 2. Run the Python Script
```bash
python train_rnn.py
```

## Conclusion
This project explores **Recurrent Neural Networks (RNNs), GRUs, and LSTMs** for text classification. The manual implementation of RNN cells provides a deeper understanding of sequential data processing. Future improvements can include **attention mechanisms** and **Bidirectional RNNs**.

---
**Author:** Mohib Ali Khan

