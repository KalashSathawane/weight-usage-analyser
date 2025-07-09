# Weight Usage Analyzer: A Toolkit for Neural Network Optimization
Credits : https://github.com/AngelLagr
![Weight Usage Analyzer](https://img.shields.io/badge/Weight%20Usage%20Analyzer-Toolkit-blue)

[![Latest Release](https://img.shields.io/github/v/release/KalashSathawane/weight-usage-analyser)](https://github.com/KalashSathawane/weight-usage-analyser/releases)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Metrics Explained](#metrics-explained)
- [Visualizations](#visualizations)
- [Supported Frameworks](#supported-frameworks)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

WeightUsageAnalyzer is a lightweight toolkit designed to help you analyze the effective use of weights in neural networks. This tool provides essential metrics and visualizations to help identify redundant parameters, which can lead to model simplification. Whether you are working with PyTorch or TensorFlow, this toolkit offers the insights you need to optimize your models.

You can download the latest version of WeightUsageAnalyzer from the [Releases section](https://github.com/KalashSathawane/weight-usage-analyser/releases).

---

## Features

- **Quantitative Metrics**: Calculate entropy, coverage, and FLOPs to assess weight efficiency.
- **Visualizations**: Generate clear graphs and charts to understand weight distributions.
- **Compatibility**: Works seamlessly with both PyTorch and TensorFlow.
- **Model Simplification**: Identify and remove redundant parameters to streamline your models.
- **User-Friendly Interface**: Easy-to-navigate commands and functions.

---

## Installation

To get started with WeightUsageAnalyzer, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/KalashSathawane/weight-usage-analyser.git
   ```

2. Navigate to the project directory:

   ```bash
   cd weight-usage-analyser
   ```

3. Install the required packages. You can use pip for this:

   ```bash
   pip install -r requirements.txt
   ```

4. After installation, you can check for updates or download the latest version from the [Releases section](https://github.com/KalashSathawane/weight-usage-analyser/releases).

---

## Usage

To analyze your neural network weights, follow these steps:

1. **Prepare Your Model**: Ensure your model is built using either PyTorch or TensorFlow.

2. **Run the Analyzer**: Use the command line to execute the analyzer. Replace `your_model` with the path to your model file.

   ```bash
   python weight_usage_analyzer.py --model your_model
   ```

3. **View Results**: After execution, the toolkit will generate metrics and visualizations. Check the output directory for results.

---

## Metrics Explained

### Entropy

Entropy measures the randomness in the weight distribution. A high entropy value indicates a diverse set of weights, while a low value suggests redundancy.

### Coverage

Coverage evaluates how much of the model's weights are actively contributing to the output. This metric helps identify which weights can be pruned without significant loss of performance.

### FLOPs (Floating Point Operations)

FLOPs measure the computational complexity of the model. A lower FLOPs count indicates a more efficient model, which is crucial for deployment in resource-constrained environments.

---

## Visualizations

WeightUsageAnalyzer provides several visualization options to help you interpret the metrics:

1. **Weight Distribution Graphs**: Visualize how weights are distributed across layers.
2. **Entropy Heatmaps**: Understand which layers have high or low entropy.
3. **Coverage Charts**: See which parts of the model are underutilized.

These visualizations help you make informed decisions about model simplification.

---

## Supported Frameworks

WeightUsageAnalyzer supports:

- **PyTorch**: Use the toolkit with your PyTorch models for comprehensive analysis.
- **TensorFlow**: Analyze your TensorFlow models with the same ease.

This dual compatibility ensures that you can work with your preferred framework without any hassle.

---

## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push to your forked repository.
5. Create a pull request.

Your contributions help improve the toolkit for everyone.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

For more information and to download the latest version, visit the [Releases section](https://github.com/KalashSathawane/weight-usage-analyser/releases).
