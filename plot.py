import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np

def display_data(loaded_data, parent, col_x, col_y, quality_col):
    x_values = [float(row[col_x]) for row in loaded_data[1:]]
    y_values = [float(row[col_y]) for row in loaded_data[1:]]
    quality_values = [float(row[quality_col]) for row in loaded_data[1:]]

    fig, ax = plt.subplots()
    scatter = ax.scatter(x_values, y_values, c=quality_values, cmap='viridis', alpha=0.7)
    ax.set_xlabel(loaded_data[0][col_x])
    ax.set_ylabel(loaded_data[0][col_y])
    ax.set_title('Quality Rating')

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(loaded_data[0][quality_col])

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=True, fill='both')

def plot_confusion_matrix(conf_matrices, parent):
    avg_conf_matrix = np.mean(conf_matrices, axis=0)

    figure, ax = plt.subplots(figsize=(10, 7))
    class_names = ['3', '4', '5', '6', '7', '8']

    sns.heatmap(avg_conf_matrix, annot=True, fmt='0.0f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(figure, parent)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=True, fill='both')
    plt.close(figure)


