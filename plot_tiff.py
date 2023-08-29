from tifffile import imread
import matplotlib.pyplot as plt

def read_plot_vol_tiff(filename):
    # Read the TIFF file
    image = imread(filename)

    # Check if image has 4 channels
    if image.shape[0] != 4:
        raise ValueError("The image does not have 4 channels!")

    # Plot each channel
    fig, axarr = plt.subplots(1, 4, figsize=(20, 5))

    channel_names = ['birefringence', 'optic axis 1', 'optic axis 2', 'optic axis 3']
    axial = 23
    for i in range(4):
        axarr[i].imshow(image[i, axial, ...], cmap='gray')
        axarr[i].axis('off')
        axarr[i].set_title(channel_names[i])

    plt.tight_layout()
    plt.show()    
    
def read_plot_img_tiff(filename):
    # Read the TIFF file
    image = imread(filename)

    # Check if image has 2 channels
    if image.shape[0] != 2:
        raise ValueError("The image does not have 4 channels!")

    # Plot each channel
    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

    channel_names = ['retardance', 'orientation']
    for i in range(2):
        axarr[i].imshow(image[i, ...], cmap='gray')
        axarr[i].axis('off')
        axarr[i].set_title(channel_names[i])

    plt.tight_layout()
    plt.show()
