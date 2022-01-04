import matplotlib.pyplot as plt


def visualize_heatmap(x, title=None):
    plt.figure()
    plt.imshow(x.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
    if title is not None:
        plt.title(title)
    plt.show()


def visualize_hist(x, num_bins=100, title=None):
    plt.figure()
    plt.hist(x.cpu().detach().numpy().reshape(-1), bins=num_bins)
    if title is not None:
        plt.title(title)
    plt.show()
