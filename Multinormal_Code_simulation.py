import numpy as np
# from numpy.linalg import inv
import matplotlib.pyplot as plt

#mean vectors
mu = np.array([2, -1]) #arbitary values

#covariance matrix
sigma = np.array([[3, 1], [1, 2]])

#defining sample size
sample_size = 1000

#finding out multivariate normal distribution given mu and sigma

dist_data = np.random.multivariate_normal(mu, sigma, sample_size)


# Scatter plot of Y1 vs Y2
plt.figure(figsize=(10, 6))
plt.scatter(dist_data[:, 0], dist_data[:, 1], alpha=0.5)
plt.title('Scatter plot of Y1 vs Y2')
plt.xlabel('Y1')
plt.ylabel('Y2')
plt.grid(True)
plt.show()


# we can move to right hand side setup now

sigma11 = sigma[0, 0]
sigma22 = sigma[1, 1]
sigma12 = sigma[0, 1]
sigma21 = sigma[1, 0]

#finding the inverse
sigma22_inv = 1/sigma22 #1 dimensional simplified data, 1 over sigma22 is sufficient

#conditional covariance

conditional_covariance = sigma11 - sigma12 * sigma22_inv * sigma21

# Y2
Y2_values = dist_data[:, 1]

#Conditional means for Y1 given Y2
mu1 = mu[0]
mu2 = mu[1]
conditional_means = mu1 + sigma12 * sigma22_inv * (Y2_values - mu2)

simulated_Y1_values = np.random.normal(conditional_means, np.sqrt(conditional_covariance ))

## Verification Process

#Getiing original Y1 values
original_Y1_values = dist_data[:, 0]

# Statistical summaries
print("Original Y1 Mean:", np.mean(original_Y1_values))
print("Simulated Y1 Mean:", np.mean(simulated_Y1_values))
print("Original Y1 Standard Deviation:", np.std(original_Y1_values))
print("Simulated Y1 Standard Deviation:", np.std(simulated_Y1_values))


#Plotting verification plot
plt.figure(figsize=(12, 6))

# Original Y1 values vs Y2
plt.subplot(1, 2, 1)
plt.scatter(Y2_values, original_Y1_values, alpha=0.5)
plt.title('Original Y1 vs Y2')
plt.xlabel('Y2')
plt.ylabel('Y1')

plt.annotate(f'Mean: {np.mean(original_Y1_values):.3f}\nStd Dev: {np.std(original_Y1_values):.3f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
             bbox=dict(boxstyle="round", fc="w"))

# Simulated Y1 values vs Y2
plt.subplot(1, 2, 2)
plt.scatter(Y2_values, simulated_Y1_values, alpha=0.5, color='red')
plt.title('Simulated Y1 vs Y2')
plt.xlabel('Y2')
plt.ylabel('Simulated Y1')

plt.annotate(f'Mean: {np.mean(simulated_Y1_values):.3f}\nStd Dev: {np.std(simulated_Y1_values):.3f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
             bbox=dict(boxstyle="round", fc="w"))

plt.tight_layout()
plt.show()

