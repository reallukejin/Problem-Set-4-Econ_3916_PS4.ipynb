# Problem-Set-4-Econ_3916_PS4.ipynb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve, KFold

# 10 shuffled folds; RMSE in dollars is easier to read than R^2
cv = KFold(n_splits=10, shuffle=True, random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
    LinearRegression(), X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv, scoring="neg_root_mean_squared_error")

train_rmse, val_rmse = -train_scores / 1000, -val_scores / 1000   # $K
tr_mean, tr_std = train_rmse.mean(axis=1), train_rmse.std(axis=1)
va_mean, va_std = val_rmse.mean(axis=1), val_rmse.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, tr_mean, "o-", label="Training RMSE")
ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.2)
ax.plot(train_sizes, va_mean, "o-", label="Validation RMSE")
ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std, alpha=0.2)
ax.set_xlabel("Training stores")
ax.set_ylabel("RMSE ($K)")
ax.set_title("Learning curve: OLS on all predictors")
ax.legend()
plt.tight_layout()
plt.show()

# GAP (validation minus training):
#   This is variance. The model fits quirks of the training stores that
#   don't carry over to new ones. If the gap is large and still shrinking
#   at the right edge, more data WILL help, because each extra store
#   pulls the two curves together.
#
# LEVEL (where both curves flatten out):
#   This is bias plus irreducible noise. If both curves meet at a high
#   error, the model is too simple or missing information. More data
#   WON'T help, because the curves are already flat. Only better
#   features or a better functional form can lower the level, and only
#   down to the noise floor (the error even the true equation would have).

gap = va_mean[-1] - tr_mean[-1]
print(f"At n={train_sizes[-1]}: train {tr_mean[-1]:.1f}K, val {va_mean[-1]:.1f}K, gap {gap:.1f}K")
print(f"Validation change over the last 3 sizes: {va_mean[-1] - va_mean[-4]:+.1f}K")
