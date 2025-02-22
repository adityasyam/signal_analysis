from PIL import Image

image_path = "./facial1_profiles.png"
image = Image.open(image_path)

# Define crop box for central portion
center_x, center_y = image.size[0] // 2, image.size[1] // 2

# Define cropping bounds (taking a smaller central region)
crop_width = 200  # Adjust as needed
crop_height = 10  # Adjust as needed

left = center_x - crop_width // 2
right = center_x + crop_width // 2

top = center_y - crop_height // 2
bottom = center_y + crop_height // 2

# Crop the image
cropped_image = image.crop((left, top, right, bottom))

# Save the cropped image
cropped_image_path = "./cropped_facial1_profiles.png"
cropped_image.save(cropped_image_path)

# Provide the download link
cropped_image_path
