# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""# Exercise 1 Drawing
Draw ellipse in the center of a black image

## Solution
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Black figure
fig = plt.figure(facecolor='black')
ax = fig.add_subplot(111)
ax.set_facecolor('black')
# Set the size of the figure
fig.set_size_inches(6, 4)

# Create an ellipse with red edge and black fill located in the center
ellipse = patches.Ellipse((0.5, 0.5), width=0.6, height=0.4, edgecolor='red', facecolor='black', transform=ax.transAxes)

# Add the ellipse to the plot
ax.add_patch(ellipse)

# Remove the axes for a cleaner look
# ax.axis('off')
plt.legend()

"""# Exercise 2
Apply to the first image (images/morpho.png) transformations to get the second and the third image

## Solution
"""

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = cv2.imread('morpho.png', cv2.IMREAD_GRAYSCALE)

# Threshold the image to make sure it's binary
_, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Define the structuring element (kernel) size for morphological operations
# kernel size adjusted to better suit specific noise characteristics in morphology image
kernel = np.ones((3,3), np.uint8)
# perform the operation based on kernel:Two steps
# Perform morphological 'opening' to remove white noise
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Perform morphological 'closing' to remove small black points
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)



# Optionally, display the images
#cv2imshow('Original Image', img)
cv2_imshow(binary)
cv2_imshow(opening)
cv2_imshow(closing)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('step2_cleaned_image.png', opening)
cv2.imwrite('step3_closed_image.png', closing)

"""# Exercise 3
Place logo (images/logo.jpg) at the image (images/track.jpg) to cover the starting cell number 2

## Solution
"""


"""Locate where the number is"""

import cv2
import numpy as np

# Load the image
image = cv2.imread('track.jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and help with edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges in the image
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edged image
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area, largest to smallest, and remove small contours that are unlikely to be the grid
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Initialize the contour which corresponds to the number "2" cell
number_two_contour = None

# Loop over the contours
for c in contours:
    # Approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # If our approximated contour has four points, then we can assume we have found our grid
    if len(approx) == 4:
        # Use boundingRect to get coordinates
        x, y, w, h = cv2.boundingRect(approx)
        # Extract the cell using the coordinates
        cell_roi = gray[y:y+h, x:x+w]
        # Check if "2" is in the cell_roi using OCR or another method
        # For now, we will just print the coordinates of the cell
        number_two_contour = approx
        break

# Check if we have found the contour
if number_two_contour is not None:
    # Draw the contour of the cell on the image
    cv2.drawContours(image, [number_two_contour], -1, (0, 255, 0), 2)

    # Print the coordinates of the vertices
    print("Coordinates of the vertices of the cell with number '2':")
    for vertex in number_two_contour:
        print(vertex[0])

    # Show the image
    cv2_imshow( image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not find the grid cell with the number '2'.")

import cv2
import numpy as np

# Load images
track_image_path = 'track.jpg'  # Replace with the path to your track image
logo_image_path = 'logo.jpg'    # Replace with the path to your logo image
track_img = cv2.imread(track_image_path)
logo_img = cv2.imread(logo_image_path)

# Manually selected points on the logo that correspond to the four corners.
# This assumes the logo is a rectangle that will be scaled and skewed to fit over cell number 2.
logo_points = np.float32([
    [0, 0],  # Top-left corner
    [logo_img.shape[1], 0],  # Top-right corner
    [logo_img.shape[1], logo_img.shape[0]],  # Bottom-right corner
    [0, logo_img.shape[0]]  # Bottom-left corner
])

# Corresponding points on the track image where the logo corners will map to.
# You need to replace these with the actual coordinates for cell number 2.
x1, y1=395, 362
x4, y4=694, 659
x3, y3=1070,  549
x2, y2= 669, 318

x4, y4=395, 362
x3, y3=694, 659
x2, y2=1070,  549
x1, y1= 669, 318
track_points = np.float32([
    [x1, y1],  # Top-left corner of cell number 2 on the track
    [x2, y2],  # Top-right corner of cell number 2 on the track
    [x3, y3],  # Bottom-right corner of cell number 2 on the track
    [x4, y4]   # Bottom-left corner of cell number 2 on the track
])

# Calculate the perspective transform matrix and warp the logo image to fit the track perspective
M = cv2.getPerspectiveTransform(logo_points, track_points)
warped_logo = cv2.warpPerspective(logo_img, M, (track_img.shape[1], track_img.shape[0]))

# Create a mask from the warped logo and combine it with the track
mask = np.all(warped_logo == [0, 0, 0], axis=2)
demask = ~mask#np.all(warped_logo != [0, 0, 0], axis=2)
warped_logo[mask] = track_img[mask]
track_img[demask] = [0,0,0]
# The logo can now be added to the track image by simply adding the images together,
# since the background of the warped logo is already transparent.
final_image = cv2.add(track_img, warped_logo)

# Save the result, adding track_img once again would introduce unnecessary lightness
cv2.imwrite('track_with_logo.jpg', final_image)

# Display the result
cv2_imshow( final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2_imshow(warped_logo)
cv2.imwrite('Output_Image.jpg',warped_logo)

"""the following is the code to detect all cells confined by painted white lines, cell of Track 3 is dismissed because it cannot form a Quadrilateral, and is obviously not relevant to our task. It is not difficult to tell which bounding box Cell 2 belongs to, with it being the most prominent."""

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
# Load the image
image = cv2.imread('track.jpg')
assert image is not None, "Image not found"

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to get a binary image
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Invert the binary image
binary = cv2.bitwise_not(binary)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# List to hold coordinates of grid cells
grid_cells = []

# Loop over the contours
for contour in contours:
    # Approximate the contour to a polygon
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # The polygon should have 4 vertices if it is a rectangle (our grid cell)
    if len(approx) == 4:
        # Compute the bounding box of the contour and use it to draw a rectangle
        x, y, w, h = cv2.boundingRect(approx)

        # Assuming a reasonable grid size (adjust these values based on your image)
        if w > 60 and h > 60:
            grid_cells.append(approx)
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

# Print the coordinates of the vertices of each grid cell
for idx, cell in enumerate(grid_cells):
    print(f"Grid Cell {idx + 1}:")
    for point in cell:
        print(point[0])

# Show the image with the detected grid cells
cv2_imshow( image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image with detected grid cells
cv2.imwrite('grid_cells_detected.jpg', image)
