# CS771: Machine Learning Major Assignment

This repository contains the implementation, derivations, and experimental results for the CS771: Machine Learning Major Assignment at IIT Kanpur[cite: 1]. The project covers semi-parametric kernel regression and delay recovery for Physical Unclonable Functions (PUFs)[cite: 1].

---

## Project Overview

### Problem 1.1: Semi-Parametric Regression via Custom Kernel (K~)

This problem focuses on modeling video hiring data using a semi-parametric regression framework[cite: 1]:
y = w(z)x + b
where x in R represents video length, z in R^2 represents video popularity and difficulty, b is the bias term, and w(z) belongs to an RKHS with a polynomial kernel K(z1, z2) = (z1^T z2 + c)^d[cite: 1].

Key Findings & Derivation:
* Augmented Kernel Formula: To convert the semi-parametric model into a pure Kernel Ridge Regression setup, we derived the augmented kernel[cite: 1]:
  K~((x1, z1), (x2, z2)) = x1 * x2 * (z1^T z2 + c)^d + 1
* Hyperparameter Tuning: Evaluated using 4-fold cross-validation across polynomial degrees d in {1, 2, 3, 4, 5} and coefficients c in {0, 0.1, 0.5, 1, 5}[cite: 1].
* Optimal Setup: d* = 2, c* = 0.5 achieving Test R^2 = 0.9699 with an execution time under 0.76s[cite: 1].

---

### Problem 1.2: Delay Recovery by Inverting a XOR Arbiter PUF

This task focuses on recovering 8 non-negative physical delay vectors (a, b, c, d, p, q, r, s) in R_{>=0}^32 for a 2-XOR Arbiter PUF (K=32 stages) given its flattened linear model w in R^1089 (w = u (X) v)[cite: 1].

Inversion Pipeline:
1. De-Kronecker Step via SVD: Reshape vector w in R^1089 into matrix W in R^{33 x 33}[cite: 1]. Compute the rank-1 SVD (W = U * Sigma * V^T) to extract single-PUF vectors u_est and v_est[cite: 1].
2. Single-PUF Inversion: Extract stage parameters (alpha_i, beta_i) and compute delay differences Delta_1 = alpha_i + beta_i and Delta_2 = alpha_i - beta_i[cite: 1].
3. Non-Negative Delay Construction: Reconstruct non-negative physical delays using constructive max operators:
   x_i = max(Delta_1, 0), y_i = max(-Delta_1, 0), z_i = max(Delta_2, 0), t_i = max(-Delta_2, 0)

---

## Experimental Results

### Problem 1.1 Performance Metrics

| Parameter | Optimal Value |
| :--- | :--- |
| Best Degree (d*) | 2[cite: 1] |
| Best Coefficient (c*) | 0.5[cite: 1] |
| Mean CV R^2 | 0.970871[cite: 1] |
| Test Set R^2 | 0.9699[cite: 1] |
| Training Time | 0.7567 s[cite: 1] |

### Problem 1.2 Delay Recovery Performance

| Metric | Measured Value |
| :--- | :--- |
| Average Decode Time | < 1 ms per model (Overall mean: ~0.39 ms)[cite: 1] |
| Reconstruction Error (||w - w_hat||_2) | O(10^-16) (Overall mean: 4.29e-16)[cite: 1] |

---

## Repository Structure

.
├── Problem1_Semiparametric/
│   ├── kernel_ridge.py      # Custom Kernel Ridge implementation
│   └── grid_search.py       # Cross-validation & hyperparameter tuning
├── Problem2_PUF_Inversion/
│   ├── decode.py            # De-Kronecker SVD & delay recovery (my_decode)
│   └── evaluate.py          # Reconstruction evaluation script
├── docs/
│   └── CS771_Report.pdf     # Full theoretical derivations and report
└── README.md

---

## Usage & Installation

### Requirements

* Python 3.8+
* numpy
* scikit-learn
* matplotlib / seaborn

pip install numpy scikit-learn matplotlib seaborn

### Running the XOR-PUF Inversion Algorithm

from decode import my_decode

# w is the 1089-dimensional linear model vector
delays = my_decode(w)

# Output contains 8 non-negative delay vectors: (a, b, c, d, p, q, r, s)
a, b, c, d, p, q, r, s = delays

---

## References

1. Rührmair, U., et al. (2010). Modeling attacks on physical unclonable functions. ACM CCS[cite: 1].
2. Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.)[cite: 1].
3. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer[cite: 1].
