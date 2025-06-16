#%% Importing libraries and document
import numpy as np
import xlrd
from collections import Counter
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

document = pd.read_excel("model_math_data.xlsx")

# %% Cleaning the dataframe
df = document.copy()
df = df.rename(columns={"Sværhedsgrad (1-3)": "Difficulty"})
df_clean = df.drop("id", axis=1)

# %% Making the datamatrix and classmatrix
X_raw = df_clean.drop("Difficulty",axis=1)
y_raw = df_clean["Difficulty"]

X = X_raw.values
N, M = X.shape
C = len(np.unique(y_raw))

n_models = X.shape[1]

acc_list=[]

for i in range(n_models):
    acc = np.unique_counts(X[:, i])
    acc_list.append((acc[1][1]/70))


fig, ax = plt.subplots(figsize=(8, 5))

# `Set2`, `Pastel1`, `tab10`, etc. all give pleasant, contrasting colours
colors = plt.cm.Set2(range(len(acc_list)))
bars = ax.bar(range(len(acc_list)), acc_list, color=colors, edgecolor="black")

ax.set_xticks(range(len(acc_list)))
ax.set_xticklabels([f"Model {i+1}" for i in range(len(acc_list))])

ax.set_ylabel("Accuracy")
ax.set_title("Accuracies of 4 Models")

ax.set_ylim(0, 1)                       # stick with proportions 0–1
ax.set_yticks(np.linspace(0, 1, 11))    # nice 0-10-…-100 % grid
ax.grid(axis="y", linestyle="--", alpha=0.6)

for rect, value in zip(bars, acc_list):  # add “85 %” labels, etc.
    ax.text(rect.get_x() + rect.get_width()/2,
            value + 0.02,
            f"{value:.0%}",
            ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()



#%%Plots with Difficulties 
# ------------------------------------------------------------------
# Masks and per-difficulty accuracies
# ------------------------------------------------------------------
difficulties   = [1, 2, 3]                       # expected values
n_models       = X.shape[1]

acc_per_diff = np.zeros((len(difficulties), n_models))   # rows=difficulty, cols=models

for d_idx, d in enumerate(difficulties):
    mask = (y_raw == d).values                  # Boolean mask for Difficulty==d
    if mask.sum() == 0:                         # safeguard: no questions of this diff?
        continue
    acc_per_diff[d_idx] = X[mask].mean(axis=0)  # mean of 0/1 → accuracy

# ------------------------------------------------------------------
# Grouped-bar chart
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

bar_w      = 0.18
x_base     = np.arange(len(difficulties))       # positions for Difficulty groups
palette    = plt.cm.Set2(range(n_models))       # distinct, pleasant colours

for m in range(n_models):
    ax.bar(x_base + m*bar_w,
           acc_per_diff[:, m],
           width=bar_w,
           color=palette[m],
           edgecolor="black",
           label=f"Model {m+1}")

# Cosmetics
ax.set_xticks(x_base + bar_w*(n_models-1)/2)
ax.set_xticklabels([f"Difficulty {d}" for d in difficulties])
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy by Question Difficulty")
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Models", bbox_to_anchor=(1.02, 1), loc="upper left")

# Annotate bars with “83 %”, “71 %”, …
for d_idx in range(len(difficulties)):
    for m in range(n_models):
        h = acc_per_diff[d_idx, m]
        ax.text(x_base[d_idx] + m*bar_w,
                h + 0.02,
                f"{h:.0%}",
                ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.show()

#%% McNemar
# ------------------------------------------------------------
# Pair-wise McNemar tests with the dtuimldmtools function
# ------------------------------------------------------------
from dtuimldmtools import mcnemar

N, n_models = X.shape
y_true = np.ones(N, dtype=int)          # dummy "all correct" ground truth
alpha  = 0.05

p_mat   = np.eye(n_models)              # 1s on the diagonal
theta   = np.zeros((n_models, n_models))
ci_L    = np.zeros((n_models, n_models))
ci_U    = np.zeros((n_models, n_models))

for i in range(n_models):
    for j in range(i + 1, n_models):
        thetahat, CI, p = mcnemar(y_true, X[:, i], X[:, j], alpha=alpha)

        p_mat[i, j] = p_mat[j, i] = p
        theta[i, j] = theta[j, i] = thetahat
        ci_L[i, j]  = ci_L[j, i]  = CI[0]
        ci_U[i, j]  = ci_U[j, i]  = CI[1]

# -------- tidy tables for inspection -------------------------
labels  = [f"Model {k+1}" for k in range(n_models)]
p_df    = pd.DataFrame(p_mat, index=labels, columns=labels).round(4)
theta_df = pd.DataFrame(theta, index=labels, columns=labels).round(3)

print("\nMcNemar p-values  (p < 0.05 ⇒ significant difference)")
print(p_df)
print("\nEstimated θ̂  (difference in accuracy)  -- positive means row model better")
print(theta_df)


        


# %%
