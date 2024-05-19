import numpy as np
# import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import os

SEL_PER_MNI = "exp_data/sel_per_mni.csv"
ABD_PER_MNI = "exp_data/abd_per_mni.csv"
ABD_ABD_MNI = "exp_data/abd_abd_mni.csv"

# Function to append data to the CSV file
def append_to_csv(filename, list_data):
    if os.path.exists(filename):
        # Load existing data and make sure it is treated as 2D
        existing_data = np.loadtxt(filename, delimiter=',')
        if existing_data.ndim == 1:
            # Handle the case where there is only one row in the file
            existing_data = existing_data.reshape(1, -1)
        # Append the new list as a new row
        new_data = np.vstack((existing_data, list_data))
    else:
        # Create new data array with the new list
        new_data = np.array([list_data])

    np.savetxt(filename, new_data, delimiter=',', fmt='%s')


def line_chart(filename, x_label='X Axis', y_label='Y Axis', title='Line Chart', save_path=None):
    # Ensure that data is a numpy array
    data = np.loadtxt(filename, delimiter=',')
    
    # Limit to the first 3 elements if there are more than 3 items
    # if data.shape[0] > 5:
    data = data[[-1,-2,-4]]
    print(len(data))
    
    # Generate distinct colors and line styles for up to 3 lines
    colors = ['b', 'g', 'r']  # blue, green, red
    linestyles = ['-', '--', '-.']
    
    plt.figure(figsize=(10, 6))

    labels=[
        'pretrain size 0',
        'pretrain size 100',
        'pretrain size 200'
    ]
    
    for idx, series in enumerate(data):
        if idx < len(colors):  # Safety check for colors and styles arrays
            plt.plot(range(len(series)), series, 
                     color=colors[idx], 
                     linestyle=linestyles[idx], 
                     label=labels[idx])

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Add legend
    plt.legend()
    if save_path:
        plt.savefig(save_path, format=save_path.split('.')[-1], dpi=300)
        print(f"Chart saved to {save_path}")
    else:
        plt.show()


def draw():
    line_chart(ABD_ABD_MNI, "EPOCH", "Accuracy", "Abduction Accuracy", "assets/charts/abd_abd_mni.png")
    line_chart(ABD_PER_MNI, "EPOCH", "Accuracy", "Perception Accuracy", "assets/charts/abd_per_mni.png")
    # line_chart(SEL_PER_MNI, "EPOCH", "Accuracy", "select", "assets/charts/sel_per_mni.png")

if __name__ == "__main__":
    draw()
