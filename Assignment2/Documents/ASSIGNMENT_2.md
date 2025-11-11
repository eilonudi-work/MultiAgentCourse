# LSTM Exercise

# 1. Background and Goal

## 1.1 Problem Statement

Given a mixed and noisy signal composed of 4 sinusoidal frequencies, the goal is to develop an LSTM-based system that extracts the pure sinusoid of each frequency.

The task includes:

- A noisy composite signal S(t) made of 4 sinusoids with random amplitude and phase.
- The LSTM receives as input:
  - The noisy signal value at time t: S[t]
  - A one-hot vector C indicating which frequency to extract.
- The output is the clean target sinusoid.

This is a conditional regression problem.

## 1.2 The Principle

Conditional Regression:

Input → Output:

| Input                                         | Output                           |
| --------------------------------------------- | -------------------------------- |
| S[t], noisy signal                            | Target clean sinusoid Targetᵢ[t] |
| One-hot vector C (which frequency to extract) |                                  |

One-hot vector for 4 frequencies:

C = [C1, C2, C3, C4]

Example:  
If we choose frequency f₂, then:

C = [0, 1, 0, 0]

## 1.3 Usage Example

Input to the model:

(S[t], C) → Output: pure sinusoid of frequency i at time t

# 2. Dataset Creation

## 2.1 General Parameters

- Frequencies:
  - f₁ = 1 Hz, f₂ = 3 Hz, f₃ = 5 Hz, f₄ = 7 Hz
- Time range: 0–10 seconds
- Sampling rate: Fs = 1000 Hz
- Number of samples: 10,000 per signal

## 2.2. **Noisy sinusoid at time \(t\):**

- Amplitude:
  A_i(t) ~ Uniform(0.8, 1.2)

- Phase:
  φ_i(t) ~ Uniform(0, 2π)

$$
\mathrm{Sinus}^{\text{noisy}}_{i}(t) \;=\; A_i(t)\,*\sin\!\bigl(2\pi*\,f_i*\,t + \phi_i(t)\bigr)
$$

## 2.3. **Mixed signal (model input):**

$$
S(t) \;=\; \frac{1}{4}\sum_{i=1}^{4}\mathrm{Sinus}^{\text{noisy}}_{i}(t)
$$

## 2.4 Train vs. Test Split

- Training set: Random seed #1
- Test set: Random seed #2 (different amplitudes and phases)

## 3. Training Dataset Structure

The training dataset contains 40,000 rows.

Each frequency generates 10,000 samples, and since there are 4 frequencies, the total size is:

4 × 10,000 = 40,000 rows.

The input for the network is a Vector of size 5:

S[t] , C1 , C2 , C3 , C4

Where:

- S[t] = the noisy mixed signal at time t
- C1..C4 = one-hot vector indicating which frequency is being extracted

# 4. Pedagogical Notes — Internal State & Sequence Length

## 4.1 LSTM Internal State

An LSTM maintains **hidden state** \(h_t\) and **cell state** \(c_t\).

## 4.2 **Critical Implementation Requirements (Sequence Length \(L=1\))**

Use **sequence length \(L=1\)** and **reset** the internal state **between every sample**:

After each training/evaluation step on a single \((S[t], C)\), **do not reset** \((h_t, c_t)\) before moving to \(t{+}1\).

# 5. Performance Evaluation

## 5.1 Success Metrics

Training MSE:

MSE_train = (1/40000) Σ (LSTM(S_train[t], C) – Target[t])²

Test MSE:

MSE_test = (1/40000) Σ (LSTM(S_test[t], C) – Target[t])²

Generalization criterion:

MSE_test ≈ MSE_train

## 5.2 Recommended Graphs

1. For frequency f₂, plot:

   - Clean target
   - Noisy input
   - LSTM output

2. For all frequencies f₁–f₄:
   - Plot target vs. output

# 6. Assignment Summary

You must:

- Generate dataset (train + test)
- Train LSTM on (S[t], C) → Targetᵢ[t]
- Use sequence length L = 1 and reset states
- Evaluate with MSE
- Produce graphs
