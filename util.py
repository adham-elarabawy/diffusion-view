from PIL import Image
import math

def img_grid(images, grid_width, grid_height):
    # Calculate the total number of images and the grid size
    num_images = len(images)
    grid_size = grid_width * grid_height
    
    # If there are more images than grid size, raise an error
    if num_images > grid_size:
        raise ValueError("The number of images exceeds the grid size.")
    
    # Calculate the dimensions of each grid cell
    cell_width = math.ceil(max(image.width for image in images))
    cell_height = math.ceil(max(image.height for image in images))
    
    # Create a blank canvas for the grid
    grid_canvas = Image.new('RGB', (grid_width * cell_width, grid_height * cell_height))
    
    # Iterate over each image and place it in the grid
    for i, image in enumerate(images):
        # Calculate the coordinates for the current grid cell
        x = (i % grid_width) * cell_width
        y = (i // grid_width) * cell_height
        
        # Resize the image to fit the grid cell size
        resized_image = image.resize((cell_width, cell_height))
        
        # Paste the resized image onto the grid canvas
        grid_canvas.paste(resized_image, (x, y))
    
    return grid_canvas