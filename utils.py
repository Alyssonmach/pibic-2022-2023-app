from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import math

def show_tensor_images(image_tensor, num_images = 25, nrow = 5, image_name = 'images/tensor_images.png'):
    '''
    Function to display a grid of images from a given tensor of images.

    Parameters:
        image_tensor (torch.Tensor): tensor of images to display.
        num_images (int): number of images to display (default = 25).
        nrow (int): number of images per row in the grid (default = 5).
        image_name (str): file name to save the image grid as (default = 'images/tensor_images.png').
    '''

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow = nrow)
    plt.imsave(image_name, image_grid.permute(1, 2, 0).squeeze().detach().numpy())

def get_input_dimensions(z_dim, n_classes):
    '''
    Function for getting the size of the conditional input dimensions 
    from z_dim, the image shape, and number of classes.

    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        n_classes: the total number of classes in the dataset, an integer scalar
    
    Returns: 
        generator_input_dim: the input dimensionality of the conditional generator, 
                             which takes the noise and class vectors
    '''

    generator_input_dim = z_dim + n_classes
    
    return generator_input_dim

def create_gif(images_pytorch_tensor, image_location):
    '''
    Function to create a GIF animation from a given tensor of images.

    Parameters:
        images_pytorch_tensor (torch.Tensor): tensor of images to create the GIF from.
        image_location (str): file name to save the GIF animation as.
    '''

    images_numpy = list()
    for index in range(0, len(images_pytorch_tensor)):
        convert_image = Image.fromarray(np.uint8(images_pytorch_tensor[index, 0, :, :].detach().numpy() * 255))
        images_numpy.append(convert_image)
    
    images_numpy[0].save(image_location, save_all = True, 
                         append_images = images_numpy, duration = 50, loop = 0)

def get_noise(n_samples, input_dim, device = 'cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.

    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    
    return torch.randn(n_samples, input_dim, device=device)

def get_one_hot_labels(labels, n_classes):
    '''
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes),
    
    Parameters:
    labels: tensor of labels from the dataloader, size (?)
    n_classes: the total number of classes in the dataset, an integer scalar
    '''

    return F.one_hot(labels, n_classes)

def combine_vectors(x, y):
    '''
    Function for combining two vectors with shape (n_samples, ?) and (n_samples, ?).
    
    Parameters:
        x: (n_samples, ?) the first vector.
            In this assignment, this will be the noise vector of shape (n_samples, z_dim),
            but you shouldn't need to know the second dimension's size.
        y: (n_samples, ?) the second vector.
            Once again, in this assignment this will be the one-hot class vector
            with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    '''

    return torch.cat(tensors = (x.float(), y.float()), dim = 1)

def interpolate_class(first_number, second_number, z_dim, n_classes, device, gen, n_interpolation = 30, 
                      show_tensor = True):
    '''
    Function to generate interpolated images between two classes using a given generator network.

    Parameters:
        first_number (int): first class label.
        second_number (int): second class label.
        z_dim (int): size of the generator's input noise vector.
        n_classes (int): number of classes in the dataset.
        device (torch.device): device to run the generator on.
        gen (torch.nn.Module): generator network to generate the interpolated images.
        n_interpolation (int): number of interpolated images to generate (default = 30).
        show_tensor (bool): flag to display the interpolated images (default = True).

    Returns:
        torch.Tensor: tensor of interpolated images.
    '''

    interpolation_noise = get_noise(1, z_dim, device = device).repeat(n_interpolation, 1)

    first_label = get_one_hot_labels(torch.Tensor([first_number]).long(), n_classes)
    second_label = get_one_hot_labels(torch.Tensor([second_number]).long(), n_classes)

    percent_second_label = torch.linspace(0, 1, n_interpolation)[:, None]
    interpolation_labels = first_label * (1 - percent_second_label) + second_label * percent_second_label

    noise_and_labels = combine_vectors(interpolation_noise, interpolation_labels.to(device))
    fake = gen(noise_and_labels)

    if show_tensor:
        show_tensor_images(fake, num_images = n_interpolation, nrow = int(math.sqrt(n_interpolation)))
    
    return fake

def interpolate_noise(first_noise, second_noise, n_interpolation, interpolation_label, n_noise, gen, 
                      device):
    '''
    Function to generate interpolated images between two noise vectors using a given generator network.

    Parameters:
        first_noise (torch.Tensor): first noise vector.
        second_noise (torch.Tensor): second noise vector.
        n_interpolation (int): number of interpolated images to generate.
        interpolation_label (torch.Tensor): class label to use for the generated images.
        n_noise (int): size of the generator's input noise vector.
        gen (torch.nn.Module): generator network to generate the interpolated images.
        device (torch.device): device to run the generator on.
    '''
    
    percent_first_noise = torch.linspace(0, 1, n_interpolation)[:, None].to(device)
    interpolation_noise = first_noise * percent_first_noise + second_noise * (1 - percent_first_noise)

    noise_and_labels = combine_vectors(interpolation_noise, interpolation_label.to(device))
    fake = gen(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation, nrow=int(math.sqrt(n_interpolation)), 
                       image_name = f'images/{n_noise}-generated.png')

def interpolate_noise_gif(n_noise, z_dim, n_interpolation, interpolation_label, gen, device):
    '''
    This function creates a GIF of interpolated noise vectors using the provided generator network.

    Parameters:
        n_noise (int): The number of noise vectors to interpolate between.
        z_dim (int): The dimensionality of the noise vectors.
        n_interpolation (int): The number of interpolation steps between each pair of noise vectors.
        interpolation_label (torch.Tensor): A one-hot encoded tensor representing the class label to generate.
        gen (torch.nn.Module): The generator network to use.
        device (str): The device to run the network on.
    '''
    
    images_numpy = list()
    plot_noises = [get_noise(1, z_dim, device = device) for i in range(10)]
    for _, first_plot_noise in enumerate(plot_noises):
        for _, second_plot_noise in enumerate(plot_noises):
            interpolate_noise(first_plot_noise, second_plot_noise, n_interpolation, interpolation_label, 
                              n_noise, gen, device)
            images_numpy.append(Image.open(f'images/{n_noise}-generated.png').convert("RGB"))

    images_numpy[0].save('images/interpolate-noise.gif', save_all = True, 
                         append_images = images_numpy, duration = 100, loop = 0)