import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("support_tickets.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.drop_duplicates()
df["Month"] = df["Date"].dt.month_name()
df["Resolution_Level"] = np.where(
    df["Resolution_Hours"] <= 5, "Fast",
    np.where(df["Resolution_Hours"] <= 10, "Moderate", "Slow")
)

print(df.head())
print(df.info())
print(df.describe())

print("Total Tickets:", len(df))
print("Average Resolution Hours:", df["Resolution_Hours"].mean())
print("Average Customer Rating:", df["Customer_Rating"].mean())

print("\nTickets by Category:")
print(df["Category"].value_counts())

print("\nAverage Resolution by Priority:")
print(df.groupby("Priority")["Resolution_Hours"].mean())

print("\nAgent Performance:")
print(df.groupby("Agent").agg(
    Total_Tickets=("Ticket_ID", "count"),
    Average_Resolution=("Resolution_Hours", "mean"),
    Average_Rating=("Customer_Rating", "mean")
))

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Category")
plt.title("Tickets by Category")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Priority", order=["Low", "Medium", "High", "Critical"])
plt.title("Tickets by Priority")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Agent", y="Resolution_Hours", errorbar=None)
plt.title("Average Resolution Time by Agent")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Priority", y="Resolution_Hours",
            order=["Low", "Medium", "High", "Critical"])
plt.title("Resolution Time by Priority")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Resolution_Hours", bins=8, kde=True)
plt.title("Distribution of Resolution Hours")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Category", y="Customer_Rating", errorbar=None)
plt.title("Customer Rating by Category")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(
    df[["Resolution_Hours", "Customer_Rating"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Resolution Time and Customer Rating Correlation")
plt.tight_layout()
plt.show()

print("\nBusiness Insights:")
print("Most Common Category:", df["Category"].mode()[0])
print("Most Common Priority:", df["Priority"].mode()[0])
print("Fastest Average Resolution Agent:",df.groupby("Agent")["Resolution_Hours"].mean().idxmin())
print("Highest Average Customer Rating Agent:",df.groupby("Agent")["Customer_Rating"].mean().idxmax())
print("Slow Tickets:", (df["Resolution_Level"] == "Slow").sum())