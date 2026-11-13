
import pandas as pd
import matplotlib.pyplot as plt

# Load model evaluation
scores = pd.read_csv("outputs/model_scores.csv")

# =====================================
# 1. ELBOW METHOD
# =====================================

plt.figure(figsize=(9, 5))

plt.plot(
    scores["Clusters"],
    scores["WCSS"],
    marker="o",
    linewidth=2
)

plt.title("Elbow Method - K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.xticks(scores["Clusters"])
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/elbow_method.png", dpi=300)
plt.show()


# =====================================
# 2. SILHOUETTE SCORE
# =====================================

plt.figure(figsize=(9, 5))

plt.plot(
    scores["Clusters"],
    scores["Silhouette"],
    marker="o",
    linewidth=2
)

plt.title("Silhouette Score Comparison")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.xticks(scores["Clusters"])
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/silhouette_comparison.png", dpi=300)
plt.show()


# =====================================
# 3. DAVIES-BOULDIN INDEX
# =====================================

plt.figure(figsize=(9, 5))

plt.plot(
    scores["Clusters"],
    scores["Davies_Bouldin"],
    marker="o",
    linewidth=2
)

plt.title("Davies-Bouldin Index Comparison")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Davies-Bouldin Index")

plt.xticks(scores["Clusters"])
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/davies_bouldin.png", dpi=300)
plt.show()


print("\nAll evaluation graphs generated successfully!")
