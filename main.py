import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# LOAD DATA
# =====================================================

file_path = r"C:\Users\Spydy\OneDrive\Desktop\eurusdd\Data\EURUSD_1H_2020-2024.csv"

df = pd.read_csv(file_path)

# =====================================================
# TIME COLUMN
# =====================================================

df["time"] = pd.to_datetime(
    df["time"],
    format="%d-%m-%Y %H:%M"
)

# =====================================================
# TARGET
# =====================================================

df["target"] = (
    df["close"].shift(-1) > df["close"]
).astype(int)

# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Returns
df["return"] = df["close"].pct_change()

# Moving Averages
df["ma10"] = df["close"].rolling(10).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ma50"] = df["close"].rolling(50).mean()

# Volatility
df["volatility"] = df["return"].rolling(20).std()

# Candle Range
df["range"] = df["high"] - df["low"]

# Previous Candle Direction
df["prev_direction"] = (
    df["close"].shift(1) > df["open"].shift(1)
).astype(int)

# Distance from Moving Averages
df["close_ma10_diff"] = df["close"] - df["ma10"]
df["close_ma20_diff"] = df["close"] - df["ma20"]

# Momentum
df["momentum_5"] = df["close"] - df["close"].shift(5)
df["momentum_10"] = df["close"] - df["close"].shift(10)

# Range Percentage
df["range_pct"] = (
    (df["high"] - df["low"]) / df["close"]
)

# Hour Feature
df["hour"] = df["time"].dt.hour

# =====================================================
# RSI 14
# =====================================================

delta = df["close"].diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["rsi"] = 100 - (100 / (1 + rs))

# =====================================================
# CLEAN DATA
# =====================================================

df = df.dropna()

print("Rows after cleaning:", len(df))

# =====================================================
# FEATURES
# =====================================================

feature_columns = [
    "return",
    "ma10",
    "ma20",
    "ma50",
    "volatility",
    "range",
    "prev_direction",
    "tick_volume",
    "spread",
    "close_ma10_diff",
    "close_ma20_diff",
    "momentum_5",
    "momentum_10",
    "range_pct",
    "rsi",
    "hour"
]

X = df[feature_columns]
y = df["target"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

# =====================================================
# RANDOM FOREST MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

predictions = model.predict(X_test)

# =====================================================
# MODEL RESULTS
# =====================================================

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# Save feature importance
importance.to_csv(
    "../results/feature_importance.csv",
    index=False
)

# =====================================================
# BACKTEST
# =====================================================

results = pd.DataFrame(index=X_test.index)

results["prediction"] = predictions

results["close"] = df.loc[X_test.index, "close"]

results["next_close"] = (
    df.loc[X_test.index, "close"].shift(-1)
)

results["future_return"] = (
    results["next_close"]
    - results["close"]
)

results["position"] = (
    results["prediction"] * 2
) - 1

results["strategy_return"] = (
    results["future_return"]
    * results["position"]
)

results = results.dropna()

# =====================================================
# PERFORMANCE METRICS
# =====================================================

total_return = results["strategy_return"].sum()

wins = (
    results["strategy_return"] > 0
).sum()

total_trades = len(results)

win_rate = wins / total_trades

print("\n==========================")
print("BACKTEST RESULTS")
print("==========================")

print("Total Trades:", total_trades)
print("Win Rate:", round(win_rate * 100, 2), "%")
print("Total Return:", round(total_return, 5))

# Save metrics
with open("../results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {round(accuracy * 100, 2)}%\n")
    f.write(f"Total Trades: {total_trades}\n")
    f.write(f"Win Rate: {round(win_rate * 100, 2)}%\n")
    f.write(f"Total Return: {round(total_return, 5)}\n")

# =====================================================
# EQUITY CURVE
# =====================================================

results["equity_curve"] = (
    results["strategy_return"].cumsum()
)

plt.figure(figsize=(12, 6))

plt.plot(results["equity_curve"])

plt.title("EURUSD ML Strategy Equity Curve")
plt.xlabel("Trades")
plt.ylabel("Cumulative Return")
plt.grid(True)

# Save chart
plt.savefig(
    "../results/equity_curve.png",
    bbox_inches="tight"
)

plt.show()

print("\nFiles Saved Successfully:")
print("results/feature_importance.csv")
print("results/metrics.txt")
print("results/equity_curve.png")