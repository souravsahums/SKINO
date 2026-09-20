"""Numerically verify Proposition 'Which form, and only which form'.

Claim:  W2 A - A^T W2 = [S, M],  S = W2 W1^-1,  M = K^T W1 D K,  M symmetric,
for A = (W1^-1 K^T W1) D K.  And the defect vanishes for all K, rho iff W2 ~ W1.
"""
import numpy as np

rng = np.random.default_rng(0)
n = 9

for trial in range(4):
    w1 = rng.uniform(0.2, 3.0, n)          # arbitrary positive diagonal
    w2 = rng.uniform(0.2, 3.0, n)
    W1, W2 = np.diag(w1), np.diag(w2)
    K = rng.normal(size=(n, n))
    D = np.diag(rng.uniform(0.1, 2.0, n))  # diag(rho')

    Kstar = np.linalg.inv(W1) @ K.T @ W1   # adjoint in W1
    A = Kstar @ D @ K

    lhs = W2 @ A - A.T @ W2
    S = W2 @ np.linalg.inv(W1)
    M = K.T @ W1 @ D @ K
    rhs = S @ M - M @ S

    sym = np.abs(M - M.T).max()
    err = np.abs(lhs - rhs).max() / (np.abs(lhs).max() + 1e-30)
    print(f"trial {trial}:  ||LHS-[S,M]||/||LHS|| = {err:.3e}   "
          f"M asymmetry = {sym:.3e}")

    # entrywise form
    s = np.diag(S)
    ent = np.abs(lhs - (s[:, None] - s[None, :]) * M).max()
    print(f"           entrywise (s_i - s_j) M_ij residual = {ent:.3e}")

print("\n--- the iff direction ---")
w1 = rng.uniform(0.2, 3.0, n)
W1 = np.diag(w1)
K = rng.normal(size=(n, n))
D = np.diag(rng.uniform(0.1, 2.0, n))
Kstar = np.linalg.inv(W1) @ K.T @ W1
A = Kstar @ D @ K

for c in (1.0, 2.5, 7.0):                   # W2 = c W1  -> must be exact
    W2 = c * W1
    d = np.abs(W2 @ A - A.T @ W2).max() / (np.abs(W2 @ A).max())
    print(f"  W2 = {c:>4.1f} * W1   relative defect = {d:.3e}")

W2 = np.diag(rng.uniform(0.2, 3.0, n))      # not proportional -> must be O(1)
d = np.abs(W2 @ A - A.T @ W2).max() / (np.abs(W2 @ A).max())
print(f"  W2 not prop. W1   relative defect = {d:.3e}")

print("\n--- the explicit counterexample used in the proof ---")
i, j = 2, 5
e = np.eye(n)
Kc = np.outer(e[i] + e[j], e[i] + e[j])
Mc = Kc.T @ W1 @ np.eye(n) @ Kc
print(f"  M_ij = {Mc[i, j]:.6f}   predicted w_i + w_j = {w1[i] + w1[j]:.6f}")
assert abs(Mc[i, j] - (w1[i] + w1[j])) < 1e-10
print("  counterexample verified")
