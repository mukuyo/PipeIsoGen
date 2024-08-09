import os
import numpy as np
import cv2

# Define the length of the arrows in pixels
arrow_length = 50

# Define the colors of the arrows
arrow_colors = [(0,0,255), (0,255,0), (255,0,0)] # BGR format

# Path to the folder containing the RGB files
folder_path = "/home/th/ws/research/PipeIsoGen/data/output/color_raw"

# Load the camera calibration matrix
camera_matrix = np.loadtxt(os.path.join("/home/th/ws/research/PipeIsoGen/data/08_06/", "cam_K.txt"))

# Loop over all PNG files in the folder
for i, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(".png"):
        # Construct the paths for the pose file and the output file
        base_filename = os.path.splitext(filename)[0]
        pose_file = os.path.join(folder_path, "../poses/"+str(i)+".txt")
        output_file = os.path.join(folder_path, "../vizualisation/"+str(i)+".png".format(base_filename))
        print(output_file)
        # Read in the pose file
        pose = np.loadtxt(pose_file)

        # Load the RGB image
        image_file = os.path.join(folder_path, filename)
        image = cv2.imread(image_file)

        # Get image dimensions and set origin to the center
        image_height, image_width, _ = image.shape
        origin = (image_width // 2, image_height // 2)

        # Extract the translation vector from the pose matrix
        translation_vector = pose[:3, 3]

        # Convert the translation vector from meters to pixels
        translation_vector = np.matmul(camera_matrix, translation_vector)
        translation_vector = translation_vector[:2] / translation_vector[2]

        # Adjust the endpoints for the arrows based on the new origin
        end_points = [
            (int(origin[0] + arrow_length * pose[2][0]), int(origin[1] - arrow_length * pose[2][1])),
            (int(origin[0] + arrow_length * pose[0][0]), int(origin[1] - arrow_length * pose[0][1])),
            (int(origin[0] + arrow_length * pose[1][0]), int(origin[1] - arrow_length * pose[1][1]))
        ]

        # Draw the arrows on the image
        for end_point, color in zip(end_points, arrow_colors):
            image = cv2.arrowedLine(image, origin, end_point, color, thickness=2)

        # Save the image with arrows
        cv2.imwrite(output_file, image)
