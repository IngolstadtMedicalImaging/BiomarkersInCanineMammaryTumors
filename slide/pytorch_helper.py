# helper function for data visualization
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import torch

def visualize(areas: None, max_values: None, dict, **images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(10,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        # get title from the parameter names
        if areas != None and str(name) == "predicted_mask": 
            percent = areas.get("Tubular")
            if percent > 0.75: 
                point = 1
            elif percent >= 0.10 and percent <= 0.75:
                point = 2
            elif percent < 0.10: 
                point = 3
            areas = str(areas).replace("{", "")
            areas = str(areas).replace("}", "")
            areas = str(areas).replace("'", "")
            plt.title("Prediction"+ ": " + "tubule area: "+ str(percent*100)+ "%" + "-> points: "+ str(point), fontsize=15)
            plt.imshow(image, vmin= 0, vmax = len(dict)-1, cmap='viridis')
        elif max_values != None and str(name) == "predicted_mask":
            print("max: ", max_values)
            #prediction: pleo 2 -> score: 2
            if max_values == 1: 
                pleo = "pleo 1"
            if max_values == 2: 
                pleo = "pleo 2"
            if max_values == 3:
                pleo = "pleo 3"
            
            plt.title("Prediction"+ ": "+ pleo + "-> points: "+ str(max_values), fontsize=15)
            plt.imshow(image, vmin= 0, vmax = len(dict)-1, cmap='viridis')
        else: 
            plt.title(name.replace('_',' ').title(), fontsize=20)
            plt.imshow(image, vmin= 0, vmax = len(dict)-1, cmap='viridis')
    plt.show()
    
    

# Perform one hot encoding on label
def one_hot_encode(label, label_values):

    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """

    label_tensor = torch.from_numpy(label).long()
    one_hot_mask = torch.nn.functional.one_hot(label_tensor, num_classes=len(label_values)).type(torch.bool)

    return one_hot_mask.numpy()


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

