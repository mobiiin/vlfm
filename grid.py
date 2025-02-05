from PIL import Image, ImageDraw, ImageFont

def draw_grid_numbers(image_path, output_path, grid_size=(32, 32)):
    """
    Draw grid numbers on an image without drawing the grid lines.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
        grid_size (tuple): Size of the grid (rows, columns). Default is (32, 32).
    """
    # Load the image
    image = Image.open(image_path)
    width, height = image.size

    # Calculate the size of each grid cell
    cell_width = width // grid_size[0]
    cell_height = height // grid_size[1]

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Load a font (you can change the font path if needed)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)  # Use a system font
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font

    # Loop through each grid cell and write the number
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Calculate the top-left corner of the grid cell
            x = j * cell_width
            y = i * cell_height

            # Calculate the grid number (1D index)
            grid_number = i * grid_size[1] + j + 1

            # Draw the grid number at the center of the cell
            text = str(grid_number)
            # Use textbbox to get the bounding box of the text
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]  # Width of the text
            text_height = bbox[3] - bbox[1]  # Height of the text

            # Calculate the position to center the text
            text_x = x + (cell_width - text_width) // 2
            text_y = y + (cell_height - text_height) // 2

            # Draw the text on the image
            draw.text((text_x, text_y), text, fill="red", font=font)

    # Save the modified image
    image.save(output_path)
    print(f"Image with grid numbers saved to {output_path}")


# Example usage
if __name__ == "__main__":
    input_image_path = "_topdownmappp.png"  # Replace with your image path
    output_image_path = "_topdownmappp_num.png"  # Replace with your desired output path

    # Draw grid numbers on the image
    draw_grid_numbers(input_image_path, output_image_path, grid_size=(10,10))