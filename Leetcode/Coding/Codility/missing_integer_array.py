# This is a demo task.

# Write a function:

# def solution(A)

# that, given an array A of N integers, returns the smallest positive integer (greater than 0) that does not occur in A.

# For example, given A = [1, 3, 6, 4, 1, 2], the function should return 5.

# Given A = [1, 2, 3], the function should return 4.

# Given A = [−1, −3], the function should return 1.

# Write an efficient algorithm for the following assumptions:

# N is an integer within the range [1..100,000];
# each element of array A is an integer within the range [−1,000,000..1,000,000].
# Copyright 2009–2022 by Codility Limited. All Rights Reserved. Unauthorized copying, publication or disclosure prohibited.

#Idea is to check if A contains positive integers as 

def solution(A):
    seen_array = [0]*len(A)
    for value in A:
        if 0<value<=len(A):
            seen_array[value-1]=1
    
    for i in range(len(A)):
        if seen_array[i]==0:
            return i+1

    
    return len(A)+1