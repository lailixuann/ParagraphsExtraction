from matplotlib import pyplot as plt
import numpy as np
import cv2

def remove_table_image(image):
    """
    Removes regions in the input image that are likely to represent tables or images.

    Parameters:
    - image: input image.

    Returns:
    - image: image with table and image regions set to white (255).
    """
    # Set threshold values based on typical area of paragraph regions.
    threshold_area = 500
    # Set threshold values based on the maximum aspect ratio of the bounding rectangle(table).
    threshold_aspect_ratio = 10
    # Apply Canny edge detection
    edges = cv2.Canny(image, 25, 150)
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        # Check if the contour area is above the threshold
        if area > threshold_area:
            # Calculate the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Calculate the aspect ratio of the bounding rectangle
            aspect_ratio = w / float(h)
            # Check if the aspect ratio is below the threshold
            if aspect_ratio < threshold_aspect_ratio:
                # Turns the pixel in identified region to white
                image[y:y+h, x:x+w] = 255
    
    cv2.imshow("Greyscale image w/o table and image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


def load_image(image_path):
    """
    Load an image, convert it to grayscale, and remove table and image from it. Then, convert it into a bianry image.

    Parameters:
    - image_path: path to the image file

    Returns:
    - grayscale image without table and image
    - binary inverse image
    """
    # Load the image
    img = cv2.imread(image_path)
    cv2.imshow("Original Image",img)
    cv2.waitKey(0)
    # Convert the image into grayscale
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.imshow("Greyscale Image",img)
    cv2.waitKey(0)
    # Remove table and image from the image
    img = remove_table_image(img)
    # Converts the input image into binary image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Binary Image",binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img,binary


def compute_vertical_projection(binary_image):
    """
    Compute the vertical projection of a binary image.

    Parameters:
    - binary_image: binary image

    Returns:
    - vertical projection array
    """
    # Compute vertical projection of the binary image
    vertical_projection = np.sum(binary_image, axis=1)
    plt.figure()
    plt.plot(vertical_projection)
    plt.title("Vertical Projection")
    plt.show()
    return vertical_projection

def compute_horizontal_projection(binary_image):
    """
    Compute the horizontal projection of a binary image.

    Parameters:
    - binary_image: binary image

    Returns:
    - horizontal projection array
    """
    # Compute horizontal projection of the binary image
    horizontal_projection = np.sum(binary_image, axis = 0)
    plt.figure()
    plt.plot(horizontal_projection)
    plt.title("Horizontal Projection")
    plt.show()
    return horizontal_projection

def remove_padding(img,row_projection, column_projection):
    """
    Remove padding from the image based on the projections.

    Parameters:
    - img: original image
    - row_projection: vertical projection array
    - column_projection: horizontal projection array

    Returns:
    - cropped image
    - updated vertical projection array
    - updated horizontal projection array
    """
    # Calculate top padding of the image
    top_pad = -1
    while row_projection[0] == 0:
        row_projection = np.delete(row_projection, 0)
        top_pad += 1
        
    # Calculate bottom padding of the image
    bottom_pad = 0   
    while row_projection[-1] == 0:
        row_projection = row_projection[:-1]
        bottom_pad -= 1
    
    # Calculate left padding of the image   
    left_pad = -1
    while column_projection[0] == 0:
        column_projection = np.delete(column_projection, 0)
        left_pad += 1
    
    # Calculate right padding of the image           
    right_pad = 0   
    while column_projection[-1] == 0:
        column_projection = column_projection[:-1]
        right_pad -= 1
    
    # Remove all paddings of the image
    cropped_img = img[top_pad:bottom_pad, left_pad:right_pad]    
    return cropped_img, row_projection, column_projection

def detect_empty_column(column_projection):
    """
    Detect columns of zero in the column_projection array.

    Parameters:
    - column_projection: horizontal projection array

    Returns:
    - list of indices representing empty columns number
    """
    empty_column = []
    # Iterate through horizontal projection array 
    for i,j in enumerate(column_projection):
        # If sum of pixel value for the column = 0, append the column number into the empty_column list
        if j == 0:
            empty_column.append(i)
    # Append the last column number into the empty_column list
    empty_column.append(i)
    return empty_column

def detect_empty_row(row_projection):
    """
    Detect rows of zero in the row_projection array.

    Parameters:
    - row_projection: vertical projection array

    Returns:
    - list of indices representing empty rows number
    """
    empty_row = []
    # Iterate through vertical projection array 
    for i,j in enumerate(row_projection):
        # If sum of pixel value for the row = 0, append the row number into the empty_row list
        if j == 0:
            empty_row.append(i)
    # Append the last row number into the empty_row list
    empty_row.append(i)
    return empty_row

def detect_gaps(empty_index, threshold = 0):
    """
    Detect consecutive indeces in the empty_index list that exceed a specified threshold.
    
    Parameters:
    - empty_index: list of indices representing gaps
    - threshold: minimum number of consecutive indices to be considered a gap (default=0)

    Returns:
    - list of indices representing detected gaps
    """
    gaps = []
    counter = 0
    for i in range(len(empty_index)-1):
        # Check if the number in the list of empty_index is consecutive
        if empty_index[i] + 1 == empty_index[i+1]:
            counter += 1
            # Check if consecutive empty index exceed the threshold
            if counter > threshold:
                # If yes, then it would be stored as a potential gap
                gaps.append(empty_index[i])
                counter = 0
        else:
            counter = 0
    # Append the last index into the gap list
    gaps.append(empty_index[-1])
    return gaps

def slice_columns(img,empty_column):
    """
    Slice the image into columns based on detected column gaps.

    Parameters:
    - img: cropped image
    - empty_column: list of indices representing column gaps

    Returns:
    - list of column images
    """
    # Detect gap between column
    column_gap = detect_gaps(empty_column, threshold=80)
    first = 0
    columns = []
    # Iterate through the column_gap list
    for i in range(len(column_gap)):
        # Extract the column region corresponding to the column gap
        columns.append(img[:,first: column_gap[i]])
        first = column_gap[i]
    return columns

def slice_paragraphs(image, columns):        
    """
    Slice paragraphs within each column and display them.

    Parameters:
    - image: original image
    - columns: list of column images
    
    Returns:
    - no return
    """
    # Iterate through the columns list
    c_counter = 0
    for column in columns:
        c_counter += 1
        cv2.imshow("Column %d" %c_counter ,column)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Compute vertical projection of the column
        _, binary = cv2.threshold(column, 127, 255, cv2.THRESH_BINARY_INV)
        row_projection = compute_vertical_projection(binary)
        # Detect empty row of the column
        empty_row = detect_empty_row(row_projection)
        # Detect gap between paragraph
        paragraph_gap = detect_gaps(empty_row, threshold=50)
        
        paragraphs = []
        first = 0
        # Iterate through the paragraph_gap list
        for j in range(len(paragraph_gap)): 
            # Extract the paragraph region corresponding to the paragraph gap
            paragraphs.append([column[first:paragraph_gap[j]]])
            first = paragraph_gap[j]
        
        # Iterate through paragraphs list and display all the paragraph 
        p_counter = 0
        for i in range(len(paragraphs)):
            for paragraph in paragraphs[i]:
                # Check if the length of the paragraph is greater than 60 to prevent diaplaying empty paragraph
                if len(paragraph) > 60:
                    p_counter += 1
                    cv2.imshow("Column %d Paragraph %d" %(c_counter, p_counter),paragraph)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    
def main(img_path):
    """
    Main function to process the image and extract paragraphs.

    Parameters:
    - img_path: Path to the image file
    
    Returns:
    - no return
    """
    # Load the image into greyscale with preprocessing (remove table and image) done
    img,binary_image = load_image(img_path)
    # Compute vertical and horizontal projection of the image
    vertical_projection = compute_vertical_projection(binary_image)
    horizontal_projection = compute_horizontal_projection(binary_image)
    # Remove the top,bottom,left, right padding of the image, and update the projections value
    cropped_image, vertical_projection, horizontal_projection = remove_padding(img, vertical_projection, horizontal_projection)
    # Detect empty rows and columns of the processed image
    empty_column = detect_empty_column(horizontal_projection)
    # Slice the image into columns (if any)
    columns = slice_columns(cropped_image, empty_column)
    # Slice each columns into respective paragraphs and display
    slice_paragraphs(cropped_image, columns) 
  
    
# Calling the function the process the image to extract the paragraphs  
main("004.png")