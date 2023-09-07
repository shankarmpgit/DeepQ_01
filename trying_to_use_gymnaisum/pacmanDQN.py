import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import sys
from PIL import Image as im

env =   gym.make("MsPacman-v0", obs_type = 'grayscale',render_mode="human")
observation, info = env.reset(seed=123, options={})

done = False

while not done:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation.shape)
    matrix = observation
    done = terminated or truncated
    nr,nc = matrix.shape
    row_reshaped_matrix = matrix.reshape(-1,2,nc)
    summed_rows = row_reshaped_matrix.mean(axis=1)
    nr,nc = summed_rows.shape
    column_reshaped_matrix = summed_rows.reshape(nr, -1, 2)
    summed_columns = column_reshaped_matrix.mean(axis =2)
    new_matrix = summed_columns[:-19, :]
    nr,nc = new_matrix.shape
    first_row  = new_matrix[0,:]
    last_row = new_matrix[nr-1,:]

    print(new_matrix.shape)
    new_matrix = np.vstack((new_matrix,last_row))
    new_matrix = np.vstack((first_row,new_matrix))

    nr,nc =  new_matrix.shape

    row_reshaped_matrix = new_matrix.reshape(-1,2,nc)
    summed_rows = row_reshaped_matrix.mean(axis=1)
    nr,nc = summed_rows.shape
    column_reshaped_matrix = summed_rows.reshape(nr, -1, 2)
    summed_columns = column_reshaped_matrix.sum(axis =2)
    plt.imshow(summed_columns, cmap='gray')
    plt.show()
        
    

env.close()

# new_matrix = matrix[:-38, :]
# nr,nc = new_matrix.shape
# row_reshaped_matrix = new_matrix.reshape(-1, 2, nc)
# summed_rows = row_reshaped_matrix.sum(axis=1)
# nr,nc = summed_rows.shape
# print(nr,nc)
# column_reshaped_matrix = summed_rows.reshape(nr, -1, 2)
# summed_columns = column_reshaped_matrix.sum(axis =2)
# nr,nc = summed_columns.shape
nr,nc = matrix.shape
row_reshaped_matrix = matrix.reshape(-1,2,nc)
summed_rows = row_reshaped_matrix.mean(axis=1)
nr,nc = summed_rows.shape
column_reshaped_matrix = summed_rows.reshape(nr, -1, 2)
summed_columns = column_reshaped_matrix.mean(axis =2)
new_matrix = summed_columns[:-19, :]
nr,nc = new_matrix.shape
first_row  = new_matrix[0,:]
last_row = new_matrix[nr-1,:]

print(new_matrix.shape)
new_matrix = np.vstack((new_matrix,last_row))
new_matrix = np.vstack((first_row,new_matrix))

nr,nc =  new_matrix.shape

row_reshaped_matrix = new_matrix.reshape(-1,2,nc)
summed_rows = row_reshaped_matrix.mean(axis=1)
nr,nc = summed_rows.shape
column_reshaped_matrix = summed_rows.reshape(nr, -1, 2)
summed_columns = column_reshaped_matrix.sum(axis =2)
plt.imshow(summed_columns, cmap='gray')
plt.show()



# new_matrix = first_row
# np.vstack((new_matrix, last_row))


# data = im.fromarray(summed_columns)
# data.save('pic.jpg')
# image = cv2.imread('pic.jpg',0)
# _,binary_image = cv2.threshold(image,128,700, cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("Number of contours:", len(contours))
# print(nr,nc)

# np.set_printoptions(threshold=sys.maxsize)
# print(summed_columns)




