import matplotlib.pyplot as plt


def visualize_heatmap(x):
    plt.imshow(x.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
    plt.show()


def visualize_hist(x, num_bins=100):
    plt.hist(x.cpu().detach().numpy().reshape(-1), bins=num_bins)
    plt.show()
