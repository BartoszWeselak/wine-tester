import plot as plt
import neural as neuro

def display_data(loaded_data, parent):
    frame = ttk.Frame(parent)
    frame.pack(expand=True, fill="both")

    tree = ttk.Treeview(frame, show='headings')

    headers = loaded_data[0]
    tree["columns"] = tuple(range(len(headers)))
    for i, header in enumerate(headers):
        tree.column(i, anchor="center", stretch=True)
        tree.heading(i, text=header)

    data = loaded_data[1:]
    for row in data:
        tree.insert("", tk.END, values=row)

    h_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    v_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)

    tree.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

    tree.grid(row=0, column=0, sticky='nsew')
    h_scrollbar.grid(row=1, column=0, sticky='ew')
    v_scrollbar.grid(row=0, column=1, sticky='ns')

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)


def create_tabs(root, loaded_data):
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")

    data_frame = ttk.Frame(notebook)
    notebook.add(data_frame, text='Data Display')
    display_data(loaded_data, data_frame)

    plot_frame = ttk.Frame(notebook)
    notebook.add(plot_frame, text='Plot Display')

    add_plot_controls(loaded_data, plot_frame)

    confusion_matrix_frame = ttk.Frame(notebook)
    notebook.add(confusion_matrix_frame, text='Confusion Matrix')

    X_train,y_train = neuro.preprocess_data(loaded_data)

    model,conf_matrix = neuro.build_and_train_model(X_train,y_train)

    # conf_matrix = neuro.evaluate_model(model,X_train, y_train)

    plt.plot_confusion_matrix(conf_matrix, confusion_matrix_frame)
    mean = X_train.mean()
    std = X_train.std()

    predict_frame = ttk.Frame(notebook)
    notebook.add(predict_frame, text='Predict')
    create_predict_tab(predict_frame, model, mean, std)

def add_plot_controls(loaded_data, parent):
    frame = ttk.Frame(parent)
    frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    column_names = loaded_data[0]

    col_options = column_names[1:]
    col_x_var = tk.StringVar(value=col_options[7])
    col_y_var = tk.StringVar(value=col_options[0])
    quality_col_var = tk.StringVar(value=col_options[10])

    ttk.Label(frame, text="X-axis Column:").pack(side=tk.LEFT)
    ttk.OptionMenu(frame, col_x_var, *col_options).pack(side=tk.LEFT, padx=5)

    ttk.Label(frame, text="Y-axis Column:").pack(side=tk.LEFT)
    ttk.OptionMenu(frame, col_y_var, *col_options).pack(side=tk.LEFT, padx=5)

    ttk.Label(frame, text="Value Column:").pack(side=tk.LEFT)
    ttk.OptionMenu(frame, quality_col_var, *col_options).pack(side=tk.LEFT, padx=5)

    def update_plot():
        for widget in parent.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()
        plt.display_data(loaded_data, parent, column_names.index(col_x_var.get()) ,
                         column_names.index(col_y_var.get()) ,
                         column_names.index(quality_col_var.get()) )

    ttk.Button(frame, text="Update Plot", command=update_plot).pack(side=tk.LEFT, padx=5)

import tkinter as tk
from tkinter import ttk


def create_predict_tab(parent, model, mean, std):
    frame = ttk.Frame(parent)
    frame.pack(expand=True, fill="both")

    input_labels = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
    ]

    entries = []

    def is_valid_number(new_value):
        if new_value == "":
            return True
        try:
            value = float(new_value)
            return value >= 0
        except ValueError:
            return False

    validate_number = frame.register(is_valid_number)

    for label in input_labels:
        row = ttk.Frame(frame)
        row.pack(fill='x', padx=5, pady=5)
        lbl = ttk.Label(row, text=label, width=15, anchor='w')
        lbl.pack(side='left')
        entry = ttk.Entry(row, validate="key", validatecommand=(validate_number, "%P"))
        entry.pack(side='right', expand=True, fill='x')
        entries.append(entry)

    def predict():
        for entry in entries:
            if entry.get() == "":
                result_label.config(text="All fields must be filled!")
                return

        user_data = [float(entry.get()) for entry in entries]
        predicted_quality = neuro.predict_quality(model, mean, std, user_data)
        result_label.config(text=f"Expected Wine Quality: {predicted_quality}")

    predict_button = ttk.Button(frame, text="Predict", command=predict)
    predict_button.pack(pady=10)

    result_label = ttk.Label(frame, text="Expected Wine Quality: ?")
    result_label.pack(pady=10)
