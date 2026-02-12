import matplotlib.pyplot as plt

def save_fig(result, label, filename, title, xlabel, ylabel, include_legend=True):
    plt.figure(figsize=(8,5))
    plt.plot(result, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if include_legend:
        plt.legend()

    plt.savefig(filename)
    plt.close()