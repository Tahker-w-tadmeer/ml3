import matplotlib.pyplot as plt
import seaborn as sns


def plot(data, c, p, plots_per_row=2):
    column_count = len(c)
    row_count = (column_count + plots_per_row - 1) // plots_per_row
    fig, axs = plt.subplots(row_count, plots_per_row, figsize=(15, 5 * row_count))

    for i in range(column_count):
        col = c[i]
        row = i // plots_per_row
        col_num = i % plots_per_row
        if row_count > 1:
            ax = axs[row, col_num]
        else:
            ax = axs[col_num]

        p(data, col, ax)
        ax.set_title(col)

    # Hide any unused subplots
    for i in range(column_count, row_count * plots_per_row):
        if row_count > 1:
            axs[i // plots_per_row, i % plots_per_row].axis('off')
        else:
            axs[i % plots_per_row].axis('off')

    plt.tight_layout()
    plt.show()


def plot_histograms(data, c, plots_per_row=2):
    plot(data, c, lambda d, col, ax: sns.histplot(d, x=col, ax=ax, hue='smoking', kde=True), plots_per_row)


def plot_bars(data, c, plots_per_row=4):
    plot(data, c, lambda d, col, ax: sns.barplot(data=d, x=col, ax=ax, hue='smoking'), plots_per_row)
