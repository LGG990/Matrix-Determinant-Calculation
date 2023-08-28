#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Program to evaluate the determinant of a matrix using Doolittle method of LU decomposition and by brute force method.
#Generation of random matrices is plotted to show how the computation time scales with matrix size.
#The computation time for LU decomposition scales with n^3, where 'n' is the matrix size, and the brute force method
#scales with n!.
#Data sets for upto n=200 is fitted to n^3 and upto n=12 for n! using parameter estimation.
           
    
    
                                        ### LU decomposition method ###


#Imports
import numpy as np 
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time
import random as r


#Import array from text file, populate array
file = open('matrix.txt', 'r')
A = np.array([[float(j) for j in i.split(' ')] for i in file.read().split('\n')])
file.close()
    
#Calcualte the determinannt using numpy library to confirm results
def check(array,print_=True): 
    det = np.linalg.det(array)
    if print_ == True:
        print("Numpy Determinant = ",det)

#Calculates the determinant of the imported array using product of diagonal elements of U
def det_func(upper,print_=True):
    diag = 1
    n = len(upper)
    for i in range(n):
        diag *= upper[i][i]
    if print_==True:
        print("\nLU Determinant = ", diag)
    
#Prints the L and U matrices, used as a convenience in the next cell
def Print(L,U):
    print("\nL matrix is = \n",L)
    print("\nU matrix is = \n",U)


#Calculates L and U matrices
def lu_Decomposition(A,result): 
    n = len(A)
    
    #Initialize both upper and lower matrix and populate with 1
    L, U = np.zeros((n,n),dtype=float), np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(n): 
            if j <= i:
                L[i][j] = 1
            if j >= i:
                U[i][j] = 1

    for j in range(n): 
        U[0][j] = A[0][j] #Calcualte 1st row of U
        L[j][0] = A[j][0]/A[0][0] #Calcualte 1st column of L, excluding element 00

    for i in range(1,n): #Iterate elements excluding first row & column.
                         #Conditional statments check which term to calculate next
            
        if i != n-1: #Calculates diagonal elements using 'i' index, 'If' statment makes sure it's not the final element
            U[i][i] = A[i][i] - sum( L[i][t]*U[t][i] for t in range(i))

        else:  #Calculates final nn'th element
            U[n-1][n-1] = A[n-1][n-1] - sum(L[n-1][t]*U[t][n-1] for t in range(n-1))

        for x in range(i+1, n): #Calculates i'th rows of U, i+1 makes sure to exclude the diagonal 
            U[i][x] = A[i][x] - sum(L[i][t]*U[t][x] for t in range(i))

        for y in range(i+1, n): #Calculates  i'th columns of L, same use of i+1 as for the rows of U
            L[y][i] = (A[y][i] - sum(L[y][t]*U[t][i] for t in range(i)))/U[i][i]
    
    #Prints the upper and lower matrices
    if result == True:
        Print(L,U)
    
    det_func(U) #Returns determinants
    check(A)
    
#Compute LU and determinant from matrix file.
#Define start and end time of computation and print the difference, using "Time" library
print("Matrix from file = \n",A,"\n")
start = time.time()
lu_Decomposition(A,True)
end = time.time()
print("\nTime taken = ", end-start)


# In[2]:


###LU determinant calculation of random matrices upto maximum size 'n'###

#The function f(x) = ax^3 is fitted to the computation times and plotted on top, the value of the fitting parameter is shown
#in the legend

n = 200 #Define maximum matrix size 

#Create a random matrix
def randomMatrix(size):
    A = np.random.randn(size,size)
    return(A)

#Define variable to hold computation times
y=[]

#Starting from n=2, calculate the computation time for the determinant upto the maximum size
for i in range(2,n+1):
    R = randomMatrix(i)
    start = time.time()
    lu_Decomposition(R,False)
    end = time.time()
    y.append(float(end-start))

#Define variable to hold matrix dimensions
x = np.linspace(2,n,n-1)

#Define the fitting function
def trial_function(x,a):
    return a*((2/3)*x**3)


#Calculate the fitting parameter
popt, pcov = sp.optimize.curve_fit(trial_function,x,y)

#Plot results
plt.figure(figsize=(12,8))
plt.xlabel('Matrix Size',fontsize=15)
plt.ylabel('Computation Time [Seconds]',fontsize=15)
plt.grid()
plt.plot(x, trial_function(x, *popt),'b-'
         ,label='fit: a=%5.3g' % tuple(popt))
plt.scatter(x,y,color='red')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend()
plt.show()

#Upon reflection the relationship is best show with a log graph


# In[3]:


###Brute force method###

#Read array from file into a list
#Lists are used in this method to avoid needing numpy library for matrix manipulation
file = open('listMatrix.txt', 'r')
listA = [[float(j) for j in i.split(',')] for i in file.readlines()]
file.close()


#Iterates over rows of array in order to delete the specified column
def delCol(array, index): 
    for i in range(len(array)):
        array[i].pop(index)
    return array

#Deletes the specified row of the array
def delRow(array, index): 
    array.pop(index)
    return array

#Iterates over every element and copies to new list/array, for calculation of minors
def clone(array): 
    clone = [[i for i in j] for j in array]
    return clone

#Main function to calculate the determinant
def Determinant(array):
    determinant = 0 #Initialise determinant value holder
    
    if len(array) == 2: #Check if the array is a 2x2, if so, return the determinant, otherwise move on
        det = array[0][0] * array[1][1] - array[1][0] * array[0][1]
        return det

    for column in range(len(array)):#If the matrix is NOT a 2x2 then:
                                    #For every row: clone array;delete row;delete column;define the sign.
                                    #Use recursion to create the determinant value of the respective minor matrix
                                    #Recursion loops until the matrix is a 2x2
        newArray = clone(array)    
        newArray = delRow(newArray,0)
        newArray = delCol(newArray,column)
        #print("\nMinor = ",newArray) #****************************Uncomment to view all matrices generated for calculation
        plus_minus = (-1)**int(column) #Sign definition
        minor = float(Determinant(newArray))#Find the determinant of each minor matrix
        determinant += float(plus_minus * minor * array[0][column]) #Sum all the determinants multiplied by the respective
                        #column value
    return(determinant)


#Calculates computation time, prints results
print("Matrix from file = \n",listA,"\n")
start = time.time()
result = Determinant(listA)
end = time.time()
print("\nObserved determinant   = ", result) 
print("Exact determinant          =", np.linalg.det(listA))
print("Computation time = ", end-start)


# In[ ]:


###Random Matrix Brute Force Method###

#For data fitting    
    

import scipy as sp
from scipy.special import factorial

#Define maximum matrix size
#Note: Sizes ~10 will take some minutes to run
n = 12

#Random array generation of numbers between -100 and 100 of a given size
def randomList(size):
    List = [[float(r.randint(-100, 100)) for j in range(size)] for i in range(size)] 
    return List

#Define variable to hold computation times
j=[]
#Define variable to hold matrix sizes
k = np.linspace(2,n,n-1)

#Loop to caluclate computation times of all matrices upto the maximum size n
for i in range(2,n+1):
    R = randomList(i)
    start = time.time()
    Determinant(R)
    end = time.time()
    j.append(float(end-start))


#Define fitting function
def trial_function(x,a):
    return a*sp.special.factorial(x)

#Calculate function parameters
popt, pcov = sp.optimize.curve_fit(trial_function,k,j)

#Define new x-axis data variable to plot a smoother function than the matrix size data
l = np.linspace(2,n,100)

#Plot results
plt.figure(figsize=(12,8))
plt.ylabel("Computation Time [Seconds]",fontsize=15)
plt.xlabel("Matrix Size",fontsize=15)
plt.grid()
plt.plot(l, trial_function(l, *popt),'b-',
        label='fit: a=%5.3g' % tuple(popt))
plt.scatter(k,j, color='red')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend()
plt.show()


# In[ ]:




