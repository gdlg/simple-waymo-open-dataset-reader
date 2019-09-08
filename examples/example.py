# Copyright (c) 2019, Grégoire Payen de La Garanderie, Durham University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import math
import cv2
from PIL import Image
import io
import sys

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2

# Create a transformation matrix for a given label box pose.
def get_transformation_matrix(box):
    tx,ty,tz = box.center_x,box.center_y,box.center_z
    c = math.cos(box.heading)
    s = math.sin(box.heading)

    return np.array([
        [ c,-s, 0,tx],
        [ s, c, 0,ty],
        [ 0, 0, 1,tz],
        [ 0, 0, 0, 1]])

# Draw a 3D bounding from a given 3D label on a given "img". "vehicle_to_image" must be a projection matrix from the vehicle reference frame to the image space.
def draw_3d_box(img, vehicle_to_image, label, colour=(255,128,128), draw_2d_bounding_box=True):
    box = label.box

    # Extract the box size
    sl, sh, sw = box.length, box.height, box.width

    # Get the vehicle pose
    box_to_vehicle = get_transformation_matrix(box)

    # Calculate the projection from the box space to the image space.
    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)


    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5)*sl, (l-0.5)*sw, (m-0.5)*sh, 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    vertices = vertices.astype(np.int32)

    draw_2d_bounding_box = False

    if draw_2d_bounding_box:
        # Compute the 2D bounding box and draw a rectangle
        x1 = np.amin(vertices[:,:,:,0])
        x2 = np.amax(vertices[:,:,:,0])
        y1 = np.amin(vertices[:,:,:,1])
        y2 = np.amax(vertices[:,:,:,1])

        x1 = min(max(0,x1),img.shape[1])
        x2 = min(max(0,x2),img.shape[1])
        y1 = min(max(0,y1),img.shape[0])
        y2 = min(max(0,y2),img.shape[0])

        if (x1 != x2 and y1 != y2):
            cv2.rectangle(img, (x1,y1), (x2,y2), colour, thickness = 1)
    else:
        # Draw the edges of the 3D bounding box
        for k in [0, 1]:
            for l in [0, 1]:
                for idx1,idx2 in [((0,k,l),(1,k,l)), ((k,0,l),(k,1,l)), ((k,l,0),(k,l,1))]:
                    cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)
        # Draw a cross on the front face to identify front & back.
        for idx1,idx2 in [((1,0,0),(1,1,1)), ((1,1,0),(1,0,1))]:
            cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)


def display_labels_on_image(camera_calibration, image, labels, display_time = -1):
    # TODO: Handle the camera distortions
    extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4,4)
    intrinsic = camera_calibration.intrinsic

    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0  0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0]])

    # Swap the axes around
    axes_transformation = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]])

    # Compute the projection matrix from the vehicle space to image space.
    vehicle_to_image = np.matmul(camera_model, np.matmul(axes_transformation, np.linalg.inv(extrinsic)))


    # Decode the JPEG image
    img = np.array(Image.open(io.BytesIO(image.image)))

    # Draw all the groundtruth labels
    for label in labels:
        draw_3d_box(img, vehicle_to_image, label)

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(display_time)
    

# Open a .tfrecord
filename = sys.argv[1]
datafile = WaymoDataFileReader(filename)

# Generate a table of the offset of all frame records in the file.
table = datafile.get_record_table()

print("There are %d frames in this file." % len(table))

# Loop through the whole file
## and display 3D labels.
for frame in datafile:
    display_labels_on_image(frame.context.camera_calibrations[0], frame.images[0], frame.laser_labels, 10)

# Alternative: Displaying a single frame:
# # Jump to the frame 150
# datafile.seek(table[150])
# 
# # Read and display this frame
# frame = datafile.read_record()
# display_labels_on_image(frame.context.camera_calibrations[0], frame.images[0], frame.laser_labels)

# Alternative: Displaying a 10 frames:
# # Jump to the frame 150
# datafile.seek(table[150])
# 
# for _ in range(10):
#     # Read and display this frame
#     frame = datafile.read_record()
#     display_labels_on_image(frame.context.camera_calibrations[0], frame.images[0], frame.laser_labels, 10)

