#%% Importing libraries and document
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

# %% Cleaning the dataframe
document = pd.read_excel("model_math_data.xlsx")
df = document.copy()
df = df.rename(columns={"Sværhedsgrad (1-3)": "Difficulty"})
df_clean = df.drop("id", axis=1)

# %% Making the datamatrix and classmatrix
X_raw    = df_clean.drop("Difficulty", axis=1)
y_raw    = df_clean["Difficulty"]
X        = X_raw.values
N, M     = X.shape
n_models = M

#%% Plot: Overall accuracies per model 
acc_list = []
for i in range(n_models):
    counts = np.bincount(X[:, i].astype(int), minlength=2)
    acc_list.append(counts[1] / N)

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.Set1(range(n_models))
bars   = ax.bar(range(n_models), acc_list, color=colors, edgecolor="black")
ax.set_xticks(range(n_models))
ax.set_xticklabels([f"Model {i+1}" for i in range(n_models)])
ax.set_ylabel("Accuracy")
ax.set_title("Accuracies of Models")
ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 11))
ax.grid(axis="y", linestyle="--", alpha=0.6)
for rect, value in zip(bars, acc_list):
    ax.text(rect.get_x() + rect.get_width()/2,
            value + 0.02,
            f"{value:.0%}",
            ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.show()

#%% Plot: Accuracy per difficulty 
difficulties = sorted(df_clean["Difficulty"].unique())
acc_per_diff = np.zeros((len(difficulties), n_models))
for d_idx, d in enumerate(difficulties):
    mask = (y_raw == d).values
    if mask.sum() > 0:
        acc_per_diff[d_idx] = X[mask].mean(axis=0)

fig, ax = plt.subplots(figsize=(9, 5))
bar_w   = 0.18
x_base  = np.arange(len(difficulties))
palette = plt.cm.Set1(range(n_models))
for m in range(n_models):
    ax.bar(x_base + m*bar_w,
           acc_per_diff[:, m],
           width=bar_w,
           color=palette[m],
           edgecolor="black",
           label=f"Model {m+1}")
ax.set_xticks(x_base + bar_w*(n_models-1)/2)
ax.set_xticklabels([f"Difficulty {d}" for d in difficulties])
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy by Question Difficulty")
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(title="Models", bbox_to_anchor=(1.02,1), loc="upper left")
for d_idx in range(len(difficulties)):
    for m in range(n_models):
        h = acc_per_diff[d_idx, m]
        ax.text(x_base[d_idx] + m*bar_w,
                h + 0.02,
                f"{h:.0%}",
                ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.show()

#%% McNemar: pairwise tests + heatmaps + CI + Bonferroni
def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    nn = np.zeros((2,2), int)
    c1 = (yhatA - y_true)==0
    c2 = (yhatB - y_true)==0
    nn[0,0] = np.sum(c1 & c2)
    nn[0,1] = np.sum(c1 & ~c2)
    nn[1,0] = np.sum(~c1 & c2)
    nn[1,1] = np.sum(~c1 & ~c2)
    n = nn.sum()
    n12, n21 = nn[0,1], nn[1,0]
    thetahat = (n12 - n21)/n
    # Beta-CI
    E = thetahat
    Q = (n**2*(n+1)*(E+1)*(1-E)) / (n*(n12+n21) - (n12-n21)**2)
    a = (E+1)*0.5*(Q-1)
    b = (1-E)*0.5*(Q-1)
    CI = tuple(lm*2 - 1 for lm in scipy.stats.beta.interval(0.95, a=a, b=b))
    # Exact p-value
    p_val = 2 * scipy.stats.binom.cdf(min(n12,n21), n=n12+n21, p=0.5)
    return thetahat, CI, p_val

# Compute pairwise McNemar
y_true = np.ones(N, dtype=int)
p_mat   = np.eye(n_models)
theta   = np.zeros((n_models,n_models))
ci_L    = np.zeros((n_models,n_models))
ci_U    = np.zeros((n_models,n_models))

for i in range(n_models):
    for j in range(i+1, n_models):
        t_hat, (lower, upper), p = mcnemar(y_true, X[:,i], X[:,j])
        p_mat[i,j]   = p_mat[j,i]   = p
        theta[i,j]   = theta[j,i]   = t_hat
        ci_L[i,j]    = ci_L[j,i]    = lower
        ci_U[i,j]    = ci_U[j,i]    = upper

labels    = [f"Model {k+1}" for k in range(n_models)]
p_df      = pd.DataFrame(p_mat,   index=labels, columns=labels)
theta_df  = pd.DataFrame(theta,    index=labels, columns=labels)

# Heatmaps for p-values & θ̂ 
fig, axes = plt.subplots(1,2, figsize=(12,5))
cmap_blue = plt.cm.Blues

im1 = axes[0].imshow(p_df, cmap=cmap_blue, vmin=0, vmax=1)
axes[0].set(title="McNemar p-values",
            xticks=range(n_models), yticks=range(n_models),
            xticklabels=labels, yticklabels=labels)
for i in range(n_models):
    for j in range(n_models):
        axes[0].text(j, i, f"{p_df.values[i,j]:.3f}",
                     ha="center", va="center", fontsize=8)
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(theta_df, cmap=cmap_blue, vmin=-1, vmax=1)
axes[1].set(title="Estimated θ̂ (accuracy difference)",
            xticks=range(n_models), yticks=range(n_models),
            xticklabels=labels, yticklabels=labels)
for i in range(n_models):
    for j in range(n_models):
        axes[1].text(j, i, f"{theta_df.values[i,j]:.3f}",
                     ha="center", va="center", fontsize=8)
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

#%% 95% konfidensintervaller for θ̂
ci_table = pd.DataFrame([["–"]*n_models for _ in range(n_models)],
                        index=labels, columns=labels)
for i in range(n_models):
    for j in range(n_models):
        if i < j:
            ci_table.iloc[i,j] = f"[{ci_L[i,j]:.3f}, {ci_U[i,j]:.3f}]"
            ci_table.iloc[j,i] = ci_table.iloc[i,j]

print("\n95% konfidensintervaller for θ̂ (difference in accuracy):")
print(ci_table.to_string())

#%% Bonferroni-correction of p-values
alpha = 0.05
k_tests = n_models*(n_models-1)//2
alpha_bonf = alpha / k_tests
print(f"\nBonferroni-adjusted α = {alpha_bonf:.4f}")

p_adj = np.minimum(p_mat * k_tests, 1.0)
p_adj_df = pd.DataFrame(p_adj, index=labels, columns=labels).round(4)

sig_mask = p_adj < alpha
sig_df   = pd.DataFrame(sig_mask, index=labels, columns=labels)

print("\nRaw p-values:")
print(p_df.round(4).to_string())
print(f"\nBonferroni-corrected p-values (×{k_tests}):")
print(p_adj_df.to_string())
print(f"\nSignificance after Bonferroni (α={alpha}):")
print(sig_df.to_string())

#%% Heatmap for Bonferroni-corrected p-values
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(p_adj_df, cmap=cmap_blue, vmin=0, vmax=1)
ax.set(title="Bonferroni-adjusted p-values",
       xticks=range(n_models), yticks=range(n_models),
       xticklabels=labels, yticklabels=labels)
for i in range(n_models):
    for j in range(n_models):
        ax.text(j, i, f"{p_adj_df.values[i,j]:.3f}",
                ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

#%% Confusion matrices as heatmaps
cms = []
pairs = []
max_count = 0
for i in range(n_models):
    for j in range(i+1, n_models):
        nn = np.zeros((2,2), int)
        c1 = X[:,i] == 1
        c2 = X[:,j] == 1
        nn[0,0] = np.sum(c1 & c2)
        nn[0,1] = np.sum(c1 & ~c2)
        nn[1,0] = np.sum(~c1 & c2)
        nn[1,1] = np.sum(~c1 & ~c2)
        cms.append(nn)
        pairs.append((i+1, j+1))
        max_count = max(max_count, nn.max())

fig, axes = plt.subplots(int(np.ceil(len(cms)/3)), 3,
                         figsize=(12, 4*int(np.ceil(len(cms)/3))))
axes = axes.flatten()
cmap_gray = plt.cm.Blues

for idx, (nn, (i, j)) in enumerate(zip(cms, pairs)):
    ax = axes[idx]
    im = ax.imshow(nn, cmap=cmap_gray, vmin=0, vmax=max_count)
    ax.set_title(f"Model {i} vs Model {j}")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels([f"Model {j} Correct", f"Model {j} Incorrect"])
    ax.set_yticklabels([f"Model {i} Correct", f"Model {i} Incorrect"])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, nn[r,c], ha="center", va="center", fontsize=12)
fig.tight_layout()
plt.show()


# %%
from scipy.stats import chi2_contingency
# Binær klassifikation: 1 = korrekt, 0 = forkert
X = df_clean.drop(columns=["Difficulty"]).values
y = df_clean["Difficulty"].values
n_models = X.shape[1]
difficulties = sorted(np.unique(y))

# Byg kontingenstabeller og kør chi2-test for hver model
chi2_results = []
for model_idx in range(n_models):
    table = []
    for d in difficulties:
        mask = y == d
        correct = np.sum(X[mask, model_idx] == 1)
        incorrect = np.sum(X[mask, model_idx] == 0)
        table.append([correct, incorrect])
    chi2_stat, p_val, dof, expected = chi2_contingency(table)
    chi2_results.append((f"Model {model_idx+1}", chi2_stat, p_val, dof))

# Vis resultater
results_df = pd.DataFrame(chi2_results, columns=["Model", "Chi2 Stat", "p-value", "df"])
print(results_df)
# %%
z = 1.96  # For 95% CI
ci_list = []
for i in range(n_models):
    x = int(acc_list[i] * N)
    n = N
    p_tilde = (x + 1) / (n + 2)
    se_tilde = np.sqrt(p_tilde * (1 - p_tilde) / (n + 2))
    ci = (max(0, p_tilde - z * se_tilde), min(1, p_tilde + z * se_tilde))
    ci_list.append(ci)

# Som tabel
ci_df = pd.DataFrame(ci_list, columns=["Lower CI", "Upper CI"])
ci_df.index = [f"Model {i+1}" for i in range(n_models)]
ci_df["Accuracy"] = acc_list
print("\n95% konfidensintervaller med +2-metoden:")
print(ci_df.round(4))
# %%
