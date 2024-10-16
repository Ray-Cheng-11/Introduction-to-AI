import os
import cv2
import glob
import numpy as np


def load_data_small():
    """
        This function loads images form the path: 'data/data_small' and return the training
        and testing dataset. The dataset is a list of tuples where the first element is the 
        numpy array of shape (m, n) representing the image the second element is its 
        classification (1 or 0).

        Parameters:
            None

        Returns:
            dataset: The first and second element represents the training and testing dataset respectively
    """

    # Begin your code (Part 1-1)
    '''
    Line 31 ~ 35:   Defining paths to directories containing image files.
    Line 37 ~ 41:   Iterating over each pair of path and label in the list. 
                    For each path, using glob.glob(path) to retrieve a list of filenames that match the specified pattern.
                    Then, for each filename, loading the image using cv2.imread(), converting it to grayscale, 
                    and appending a tuple (img, label) to the train_data list. The label indicates the class of the image (1 for face, 0 for non-face).
    Line 43 ~ 47:   Similar to loading training data.
    Line 49:        Combining the train_data and test_data lists into a single dataset list
    '''
    path = "data/data_small"
    test_path_nonface = path + '/test/non-face/*.pgm'
    test_path_face = path + '/test/face/*.pgm'
    train_path_nonface = path + '/train/non-face/*.pgm'
    train_path_face = path + '/train/face/*.pgm'

    train_data = []
    for path, label in [(train_path_face, 1), (train_path_nonface, 0)]:
        for filename in glob.glob(path):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            train_data.append((img, label))

    test_data = []
    for path, label in [(test_path_face, 1), (test_path_nonface, 0)]:
        for filename in glob.glob(path):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            test_data.append((img, label))

    dataset = [train_data, test_data]
    # End your code (Part 1-1)
    
    return dataset


def load_data_FDDB(data_idx="01"):
    """
        This function generates the training and testing dataset  form the path: 'data/data_FDDB'.
        The dataset is a list of tuples where the first element is the numpy array of shape (m, n)
        representing the image the second element is its classification (1 or 0).
        
        In the following, there are 4 main steps:
        1. Read the .txt file
        2. Crop the faces using the ground truth label in the .txt file
        3. Random crop the non-faces region
        4. Split the dataset into training dataset and testing dataset
        
        Parameters:
            data_idx: the data index string of the .txt file

        Returns:
            train_dataset: the training dataset
            test_dataset: the testing dataset
    """

    with open("data/data_FDDB/FDDB-folds/FDDB-fold-{}-ellipseList.txt".format(data_idx)) as file:
        line_list = [line.rstrip() for line in file]

    # Set random seed for reproducing same image croping results
    np.random.seed(0)

    face_dataset, nonface_dataset = [], []
    line_idx = 0

    # Iterate through the .txt file
    # The detail .txt file structure can be seen in the README at https://vis-www.cs.umass.edu/fddb/
    while line_idx < len(line_list):
        img_gray = cv2.imread(os.path.join("data/data_FDDB", line_list[line_idx] + ".jpg"), cv2.IMREAD_GRAYSCALE)
        num_faces = int(line_list[line_idx + 1])

        # Crop face region using the ground truth label
        face_box_list = []
        for i in range(num_faces):
            # Here, each face is denoted by:
            # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
            coord = [int(float(j)) for j in line_list[line_idx + 2 + i].split()]
            x, y = coord[3] - coord[1], coord[4] - coord[0]            
            w, h = 2 * coord[1], 2 * coord[0]

            left_top = (max(x, 0), max(y, 0))
            right_bottom = (min(x + w, img_gray.shape[1]), min(y + h, img_gray.shape[0]))
            face_box_list.append([left_top, right_bottom])
            # cv2.rectangle(img_gray, left_top, right_bottom, (0, 255, 0), 2)

            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()
            face_dataset.append((cv2.resize(img_crop, (19, 19)), 1))

        line_idx += num_faces + 2

        # Random crop N non-face region
        # Here we set N equal to the number of faces to generate a balanced dataset
        # Note that we have already save the bounding box of faces into `face_box_list`, you can utilize it for non-face region cropping
        for i in range(num_faces):
            # Begin your code (Part 1-2)
            '''
            For each face, we continue to generate random crops of the image until we find a region that doesn't overlap with any of the faces. 
            This is implemented by generating a random top-left coordinate (`rand_top_left`) and a random bottom-right coordinate (`rand_bottom_right`), 
            ensuring that they have greater x and y values. 
            We then check if this randomly generated region overlaps with any of the previously detected face regions stored in `face_box_list`. 
            If the region does not overlap and its area is greater than 0, indicating a valid non-face region,
            we crop it from the grayscale image `img_gray` and append it, along with the label `0`, to the `nonface_dataset`.
            '''

            # Initialize a flag to check if the random cropping overlaps any of the face regions
            overlap = True

            # Generate random non-face regions until no overlap is found
            while overlap:
                # Generate random top-left and bottom-right coordinates for cropping
                rand_top_left = (np.random.randint(0, img_gray.shape[1] - 19), np.random.randint(0, img_gray.shape[0] - 19))
                rand_bottom_right = (rand_top_left[0] + 19, rand_top_left[1] + 19)

                # Check for overlap with any of the face regions
                overlap = False
                for face_box in face_box_list:
                    if (rand_bottom_right[0] >= face_box[0][0] and rand_top_left[0] <= face_box[1][0] and
                            rand_bottom_right[1] >= face_box[0][1] and rand_top_left[1] <= face_box[1][1]):
                        overlap = True
                        break

            # Crop the non-face region and add it to the non-face dataset
            img_crop = img_gray[rand_top_left[1]:rand_bottom_right[1], rand_top_left[0]:rand_bottom_right[0]].copy()

            # End your code (Part 1-2)

            nonface_dataset.append((cv2.resize(img_crop, (19, 19)), 0))

        # cv2.imshow("windows", img_gray)
        # cv2.waitKey(0)

    # train test split
    num_face_data, num_nonface_data = len(face_dataset), len(nonface_dataset)
    SPLIT_RATIO = 0.7

    train_dataset = face_dataset[:int(SPLIT_RATIO * num_face_data)] + nonface_dataset[:int(SPLIT_RATIO * num_nonface_data)]
    test_dataset = face_dataset[int(SPLIT_RATIO * num_face_data):] + nonface_dataset[int(SPLIT_RATIO * num_nonface_data):]

    return train_dataset, test_dataset


def create_dataset(data_type):
    if data_type == "small":
        return load_data_small()
    else:
        return load_data_FDDB()
