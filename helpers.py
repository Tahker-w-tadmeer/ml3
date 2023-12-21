import matplotlib.pyplot as plt
import seaborn as sns


def plot_histograms(data, c, plots_per_row=2):
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

        sns.histplot(data, x=col, ax=ax, hue='smoking', kde=True)
        ax.set_title(col)

    # Hide any unused subplots
    for i in range(column_count, row_count * plots_per_row):
        if row_count > 1:
            axs[i // plots_per_row, i % plots_per_row].axis('off')
        else:
            axs[i % plots_per_row].axis('off')

    plt.tight_layout()
    plt.show()



def plot_bars(data, c):
    column_count = len(c)
    row_count = (column_count + 1) // 2
    fig, axs = plt.subplots(row_count, 2, figsize=(15, 5 * row_count))

    for i in range(column_count):
        col = c[i]
        sns.barplot(data=data, x=col, ax=axs[i // 2, i % 2], hue='smoking')
        axs[i // 2, i % 2].set_title(col)

    plt.tight_layout()
    plt.show()
