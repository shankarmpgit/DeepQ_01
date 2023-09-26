import numpy as np
import random

def simple_maze(rows,columns):
    m = np.ones((rows,columns))
    return m

def less_simple_maze(rows,columns):
    m = np.ones((rows,columns))
    for row in range(rows):
        for column in range(columns):
            if row == 0: m[row,column] = -1
            if row == rows -1: m[row,column] = -1
            if column == 0: m[row,column] = -1
            if column == columns -1: m[row,column] = -1
    return m

def medium_maze(rows,columns,number_of_walls):
    m = np.ones((rows,columns))
    for row in range(rows):
        for column in range(columns):
            if row == 0: m[row,column] = -1
            if row == rows -1: m[row,column] = -1
            if column == 0: m[row,column] = -1
            if column == columns -1: m[row,column] = -1

    for i in range(number_of_walls):
        random_i = random.randint(1,rows-2)
        random_j = random.randint(1,columns-2)
        m[random_i,random_j] = -1
        #print(f'i:{random_i}, j:{random_j}')


    return m

def bernard_maze(rows =11, columns =11):
    m = np.ones((rows,columns))
    for row in range(rows):
        for column in range(columns):
            if row == 0: m[row,column] = -1
            if row == rows -1: m[row,column] = -1
            if column == 0: m[row,column] = -1
            if column == columns -1: m[row,column] = -1
   
    m[1,5] = -1
    m[2,(2,3,5,7,8)] = -1 
    m[4,(2,4,5,6,8)] =-1
    m[5,(2,10)] =-1
    m[9,5] = -1
    m[8,(2,3,5,7,8)] = -1 
    m[6,(2,4,5,6,8)] =-1  

    return m

def print_maze(m):
    print(m)


def main():
    m = medium_maze(7,7,5)
    print(m)

if __name__ == '__main__':
    main()

    

