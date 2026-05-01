# Wheel Flat Fault Classification Using Mamdani Fuzzy Inference

A Python implementation of a Mamdani fuzzy inference system for classifying wheel-flat condition severity from vibration-derived features. The project uses triangular membership functions, automatic fuzzy rule generation from training data, duplicate-rule resolution, and centroid defuzzification to classify samples as **Healthy**, **Mild**, or **Severe**.

This project is designed as a compact, explainable machine-learning / soft-computing pipeline for condition monitoring and fault diagnosis.

## Project Overview

Wheel-flat defects can cause abnormal vibration patterns in railway wheels. Instead of using a black-box classifier, this project applies fuzzy logic so the classification process remains interpretable through human-readable rules such as:

```text
IF rms_g is Medium AND kurtosis is High THEN class is Severe
```

The system uses two input features:

- `rms_g` - root mean square vibration magnitude
- `kurtosis` - statistical measure of peak sharpness / impulsiveness

The output class is:

- `0` - Healthy
- `1` - Mild wheel-flat condition
- `2` - Severe wheel-flat condition

## Key Features

- Implements a Mamdani fuzzy inference engine from scratch in Python
- Uses triangular membership functions for input and output variables
- Generates fuzzy rules automatically from labeled training data
- Resolves duplicate/conflicting rules using accumulated rule weights
- Performs fuzzy aggregation and centroid-based defuzzification
- Includes visualization scripts for dataset distribution and membership functions
- Uses separate training and testing CSV files
- Keeps the classifier interpretable through readable rule-based logic

## Repository Structure

```text
Fuzzy_mini_project2/
├── README.md
├── Fuzzy_mini_project2_report.pdf
└── FC_mini_project_code_source/
    ├── WheelFlat_Dataset1_Train (2).csv
    ├── WheelFlat_Dataset2_Test (2).csv
    ├── dataset_scatter_plot.py
    ├── membershipfunc.py
    ├── rule_generation.py
    ├── modified_rule_base.py
    ├── mamdani_inference_engine.py
    └── main.py
```

## Dataset

The project uses two CSV files:

| File | Purpose | Samples | Columns |
| --- | --- | ---: | --- |
| `WheelFlat_Dataset1_Train (2).csv` | Training data for rule generation | 40 | `rms_g`, `kurtosis`, `label` |
| `WheelFlat_Dataset2_Test (2).csv` | Test data for inference | 30 | `rms_g`, `kurtosis`, `label` |

Class distribution:

| Dataset | Healthy | Mild | Severe |
| --- | ---: | ---: | ---: |
| Training | 14 | 13 | 13 |
| Testing | 10 | 10 | 10 |

## Fuzzy System Design

### Input 1: `rms_g`

The `rms_g` feature is represented using three triangular fuzzy sets:

| Fuzzy Set | Center `m` | Width `b` |
| --- | ---: | ---: |
| Low | 0.385 | 0.228 |
| Medium | 0.613 | 0.228 |
| High | 0.841 | 0.228 |

### Input 2: `kurtosis`

The `kurtosis` feature is represented using three triangular fuzzy sets:

| Fuzzy Set | Center `m` | Width `b` |
| --- | ---: | ---: |
| Low | 3.90 | 1.58 |
| Medium | 5.48 | 1.58 |
| High | 7.06 | 1.58 |

### Output: Class Label

The output class is represented using three triangular fuzzy sets:

| Class | Numeric Label | Center `m` | Width `b` |
| --- | ---: | ---: | ---: |
| Healthy | 0 | 0 | 1 |
| Mild | 1 | 1 | 1 |
| Severe | 2 | 2 | 1 |

### Triangular Membership Function

The membership degree is calculated as:

```python
t = max(1 - abs(x - m) / b, 0)
```

Where:

- `x` is the input value
- `m` is the center of the triangle
- `b` is the triangle width
- `t` is the membership degree between 0 and 1

## Inference Workflow

The full classification workflow is:

1. Load training data from `WheelFlat_Dataset1_Train (2).csv`
2. Compute membership degrees for `rms_g`, `kurtosis`, and `label`
3. Generate one fuzzy rule per training sample
4. Assign each rule a weight based on input membership strength
5. Merge duplicate antecedent rules using class-wise accumulated weights
6. Load test data from `WheelFlat_Dataset2_Test (2).csv`
7. Evaluate all Mamdani rules for each test sample
8. Aggregate output membership functions
9. Defuzzify the final output using the centroid method
10. Round the defuzzified value to the nearest class label


## Current Limitations

- The dataset is small, so results should be treated as a proof of concept rather than a production-ready diagnostic model.
- Membership function parameters are manually defined instead of optimized automatically.
- The current implementation prints results directly instead of returning structured metrics such as accuracy, precision, recall, or confusion matrix.
- The code is written for educational clarity and can be further modularized for production use.


## Author

Prepared as a mini project for Fuzzy Control coursework.

If you use this project in a resume or portfolio, consider adding your name, university, course, and a short project demo screenshot here.
