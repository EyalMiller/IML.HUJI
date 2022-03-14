from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import matplotlib.pyplot as plt
import numpy as np

# Q2:
''''''

# Q3:
'''pdf = my_gauss.pdf(dataset)
# plt.plot(x, dataset_plot)
plt.scatter(dataset, np.zeros(NUM))
plt.scatter(dataset, pdf, edgecolors='red')

plt.show()'''

# Q4:
'''multi_gauss = MultivariateGaussian()
multi_mu = [0, 0, 4, 0]
cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
multi_dataset = np.random.multivariate_normal(multi_mu, cov, NUM)
multi_gauss.fit(multi_dataset)

mu_range_num = 200
f1 = np.linspace(-10, 10, mu_range_num)
f3 = np.linspace(-10, 10, mu_range_num)
all_mu = np.zeros((mu_range_num, mu_range_num, 4))
results = np.zeros((mu_range_num, mu_range_num))
for i in range(mu_range_num):
    all_mu[i, :, 0] = f1
    all_mu[:, i, 2] = f3
for i in range(mu_range_num):
    for j in range(mu_range_num):
        results[i][j] = MultivariateGaussian.log_likelihood(all_mu[i][j], cov, multi_dataset)
        print(i, j)

fig, my_plot = plt.subplots()
image = my_plot.imshow(results, extent=[-10, 10, -10, 10])

# Show all ticks and label them with the respective list entries
plt.show()
max_value = np.amax(results)
max_index = np.where(results == max_value)
print(max_index[0], max_index[1])
print(max_value)'''


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    print("Q1:")
    my_gauss = UnivariateGaussian()
    dataset = np.random.normal(MU, VAR, NUM)
    # x = np.linspace(5, 15, NUM)
    my_gauss.fit(dataset)
    print("(" + str(my_gauss.mu_) + ", " + str(my_gauss.var_) + ")")

    print("---------------------------------------------------------")

    # Question 2 - Empirically showing sample mean is consistent
    print("Q2:")
    sample_count = np.linspace(10, 1000, num=100).astype(int)
    mean_estimators = np.array([])
    for n_samples in sample_count:
        new_gauss = UnivariateGaussian()
        new_gauss.fit(np.random.choice(dataset, n_samples))
        mean_estimators = np.append(mean_estimators, new_gauss.mu_)

    plt.plot(sample_count, np.abs(mean_estimators - 10))
    plt.title("Error of estimator as a function of the number of samples")
    plt.ylabel("Abs(EstimatedMean - RealMean)")
    plt.xlabel("number of samples")
    plt.show()

    print("---------------------------------------------------------")

    # Question 3 - Plotting Empirical PDF of fitted model
    print("Q3:")
    pdf = my_gauss.pdf(dataset)
    plt.title("the fitted gaussian and the empirical pdf")
    plt.scatter(dataset, np.zeros(NUM))
    plt.scatter(dataset, pdf, color='red')
    plt.legend(["fitted gaussian", "samples"])
    plt.xlabel("sample value")
    plt.ylabel("pdf")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    print("Q4:")
    multi_gauss = MultivariateGaussian()
    multi_mu = [0, 0, 4, 0]
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    multi_dataset = np.random.multivariate_normal(multi_mu, cov, NUM)
    multi_gauss.fit(multi_dataset)
    print(multi_gauss.mu_)
    print(multi_gauss.cov_)

    print("---------------------------------------------------------")

    # Question 5 - Likelihood evaluation
    print("Q5:")
    f1 = np.linspace(-10, 10, MU_SAMPLES)
    f3 = np.linspace(-10, 10, MU_SAMPLES)
    f3 = np.flip(f3)  # so that the graph will start at (-10, -10) and end in (10, 10)
    all_mu = np.zeros((MU_SAMPLES, MU_SAMPLES, 4))
    results = np.zeros((MU_SAMPLES, MU_SAMPLES))
    for i in range(MU_SAMPLES):
        # keep in mind - I intentionally swapped f1 and f3 so that they will be on the correct
        # axis according to the question
        all_mu[i, :, 0] = f1
        all_mu[:, i, 2] = f3
    for i in range(MU_SAMPLES):
        for j in range(MU_SAMPLES):
            results[i][j] = MultivariateGaussian.log_likelihood(all_mu[i][j], cov, multi_dataset)
    _, my_plot = plt.subplots()
    my_plot.imshow(results, extent=[-10, 10, -10, 10])
    plt.title("heatmap of log-likelihood as a function of f1, f3")
    plt.xlabel("f1")
    plt.ylabel("f3")
    plt.show()
    print("I can learn from the graph that the best values are near the original values\n"
          "we used to fit the model - 0 for f1 and 4 for f3")
    print("---------------------------------------------------------")

    # Question 6 - Maximum likelihood
    print("Q6:")
    max_value = np.amax(results)
    print("Max Value:", max_value.round(3))
    max_index_f1 = np.where(results == max_value)[1][0]
    max_index_f3 = np.where(results == max_value)[0][0]
    print("Could be find in indices:", max_index_f1, max_index_f3)
    print("It means that the best model is when:")

    # I used the linspace and adjusted the index so that I could find the original value,
    # because the center is 100 and the length is 10:
    best_f1 = (max_index_f1 - 100) / 10
    best_f3 = (max_index_f3 - 100) / 10
    print("f1 =", best_f1)
    print("f3 =", best_f3)


if __name__ == '__main__':
    FEATURES = 4
    NUM = 250
    MU = 10
    VAR = 1
    MU_SAMPLES = 200
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
