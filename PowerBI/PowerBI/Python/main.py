import pandas as pd
import yfinance as yf
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# 1. FRED API KEY
# =============================
fred = Fred(api_key="1fb2df92a13a7551a8f402b91cbeedf3")

# =============================
# 2. DOWNLOAD USD/INR
# =============================
usd_inr = yf.download("INR=X", start="2010-01-01", interval="1d")

# Fix MultiIndex issue (new yfinance versions)
if isinstance(usd_inr.columns, pd.MultiIndex):
    usd_inr.columns = usd_inr.columns.get_level_values(0)

usd_inr = usd_inr[['Close']]
usd_inr.rename(columns={'Close': 'USD_INR'}, inplace=True)

# =============================
# 3. DOWNLOAD BRENT OIL
# =============================
brent = yf.download("BZ=F", start="2010-01-01", interval="1d")

if isinstance(brent.columns, pd.MultiIndex):
    brent.columns = brent.columns.get_level_values(0)

brent = brent[['Close']]
brent.rename(columns={'Close': 'Brent_Oil'}, inplace=True)

# =============================
# 4. DOWNLOAD DOLLAR INDEX
# =============================
dxy = yf.download("DX-Y.NYB", start="2010-01-01", interval="1d")

if isinstance(dxy.columns, pd.MultiIndex):
    dxy.columns = dxy.columns.get_level_values(0)

dxy = dxy[['Close']]
dxy.rename(columns={'Close': 'Dollar_Index'}, inplace=True)

# =============================
# 5. FRED DATA
# =============================
us_interest = fred.get_series("FEDFUNDS").to_frame(name="US_Interest_Rate")
india_cpi = fred.get_series("INDCPIALLMINMEI").to_frame(name="India_CPI")

# Convert FRED index to datetime
us_interest.index = pd.to_datetime(us_interest.index)
india_cpi.index = pd.to_datetime(india_cpi.index)

# =============================
# 6. CONVERT ALL TO MONTHLY (IMPORTANT FIX)
# =============================
usd_inr.index = pd.to_datetime(usd_inr.index)
brent.index = pd.to_datetime(brent.index)
dxy.index = pd.to_datetime(dxy.index)

# NEW pandas uses "ME" instead of "M"
usd_inr = usd_inr.resample("ME").mean()
brent = brent.resample("ME").mean()
dxy = dxy.resample("ME").mean()
us_interest = us_interest.resample("ME").mean()
india_cpi = india_cpi.resample("ME").mean()

# =============================
# 7. MERGE DATASETS
# =============================
df = usd_inr.merge(brent, left_index=True, right_index=True, how="inner")
df = df.merge(dxy, left_index=True, right_index=True, how="inner")
df = df.merge(us_interest, left_index=True, right_index=True, how="inner")
df = df.merge(india_cpi, left_index=True, right_index=True, how="inner")

# Drop missing values
df = df.dropna()

# Save final dataset
df.to_csv("macro_combined_dataset.csv")

print("Dataset successfully created!")
# avg
avg_usd_inr = df["USD_INR"].mean()
print(avg_usd_inr)
df = df.copy()

# Ensure index is datetime
df.index = pd.to_datetime(df.index)

# Create YearMonth column
df["YearMonth"] = df.index.strftime("%Y-%m")

plt.figure(figsize=(10,5))
plt.plot(df.index, df['USD_INR'])
plt.title("USD/INR Trend Over Time")
plt.xlabel("Year")
plt.ylabel("Exchange Rate")
plt.show()
df['Volatility_6M'] = df['USD_INR'].rolling(window=6).std()

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Volatility_6M'])
plt.title("6-Month Rolling Volatility of USD/INR")
plt.show()
macro_cols = [
    "USD_INR",
    "India_CPI",
    "US_Interest_Rate",
    "Brent_Oil",
    "Dollar_Index"
]

macro_df = df[macro_cols]
correlation_matrix = macro_df.corr()
print(correlation_matrix)


plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix - Macro Variables")
plt.show()

returns_df = df[[
    "USD_INR",
    "India_CPI",
    "US_Interest_Rate",
    "Brent_Oil",
    "Dollar_Index"
]].pct_change().dropna()

movement_corr = returns_df.corr()
print(movement_corr)

df['Volatility_6M']

# Sort by highest volatility
top_volatility = df.sort_values(by="Volatility_6M", ascending=False)

# Select top 5
top_5_volatility = top_volatility.head(5)

print(top_5_volatility[[
    "YearMonth",
    "USD_INR",
    "Volatility_6M",
    "Dollar_Index",
    "Brent_Oil",
    "US_Interest_Rate"
]])
cnf = df.loc["2012":"2014", ["USD_INR"]]
print(cnf)


rolling_corr_dxy = (
    returns_df["USD_INR"]
    .rolling(6)
    .corr(returns_df["Dollar_Index"])
)

rolling_corr_oil = (
    returns_df["USD_INR"]
    .rolling(6)
    .corr(returns_df["Brent_Oil"])
)




plt.figure(figsize=(12,6))

# Plot rolling correlations
plt.plot(rolling_corr_dxy.index, rolling_corr_dxy, linewidth=2, label="USD_INR vs Dollar Index")
plt.plot(rolling_corr_oil.index, rolling_corr_oil, linewidth=2, label="USD_INR vs Brent Oil")

# Add reference line at zero
plt.axhline(0, linestyle="--")

# Labels and title
plt.xlabel("Date")
plt.ylabel("Rolling Correlation (6M)")
plt.title("Rolling 6-Month Correlation of USD/INR with Macro Drivers")

# Legend
plt.legend()

# Improve layout
plt.tight_layout()

plt.show()
# Exporting into CSV
# Final cleaning export
df_export = df.reset_index()
df["RollingCorr_DXY"] = rolling_corr_dxy
df["RollingCorr_Oil"] = rolling_corr_oil

'''df_export = df.reset_index()
df_export.to_csv("macro_combined_dataset.csv", index=False)'''
corr_matrix = movement_corr.reset_index()
corr_matrix.to_csv("correlation_matrix.csv", index=False)


print("CSV file successfully created and ready for Power BI!")