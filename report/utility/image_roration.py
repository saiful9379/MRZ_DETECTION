import os
from PIL import Image

def rotate_image(image, angle):
    # Open the image file
    

    # Rotate the image
    rotated_image = image.rotate(angle, expand=True)

    # Display the rotated image
    # rotated_image.show()
    return rotated_image

# Specify the image path


# Rotate the image by -45 degrees
if __name__ == "__main__":
    image_path = "./data/training/03-02-2023-12-14-081977_phase_1_phase_1.png"
    output_dir = "./data"
    rotation_list = [45, 90, 135, 180, 215, 250, 290, 315, 360]
    for angle in rotation_list:
        file_name = os.path.basename(image_path)

        image = Image.open(image_path)

        width, height = image.size

        print("before height and width ", width, height )

        r_image = rotate_image(image, angle)

        rwidth, rheight = r_image.size

        print("after height and width ", rwidth, rheight )

        output_file_path = os.path.join(output_dir, file_name[:-4]+"_angle_"+str(angle)+".png")
        r_image.save(output_file_path)




