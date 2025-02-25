import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data: Number of Louvain runs and corresponding unique genes observed with ZEB2
runs = np.array([1, 2, 5, 10, 25, 50, 100, 500, 1000])
unique_genes = np.array([64, 121, 233, 424, 834, 1095, 1366, 1821, 1843])
total_genes = 1866

# Define the Michaelis-Menten function
def michaelis_menten(x, L, k):
    return L * x / (x + k)

# Initial guess for L and k; L near total genes, k maybe around 10 (adjust as needed)
initial_guess = [total_genes, 10]

# Fit the Michaelis-Menten model
popt, pcov = curve_fit(michaelis_menten, runs, unique_genes, p0=initial_guess)
L_fit, k_fit = popt
print(f"Fitted parameters: L = {L_fit:.2f}, k = {k_fit:.2f}")

# Generate smooth curve for plotting
x_fit = np.linspace(0, runs[-1], 1000)
y_fit = michaelis_menten(x_fit, L_fit, k_fit)

# Plot the data and the fitted curve
plt.figure(figsize=(10, 6))
plt.plot(runs, unique_genes, 'o', label="Observed Data", linestyle="-")
plt.plot(x_fit, y_fit, '-', label=f"Fit: L = {L_fit:.1f}, k = {k_fit:.1f}")
plt.xlabel("Number of Louvain Runs")
plt.ylabel("Unique Genes in Same Community as ZEB2")
plt.title("Michaelis-Menten Regression on Unique Genes vs. Louvain Runs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("unique_genes_michaelis_menten.png")
plt.show()
