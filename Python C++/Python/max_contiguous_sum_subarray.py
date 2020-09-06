
"""
The idea is to keep two counters that contain sum (max_so_far) of till an index 
and another sum (max_ending_here) including the index. If the max_so_far is less than
max_ending_here here we update the max_so_far and move forward.

Anytime when max_ending_here becomes negative we convert it to zero.
"""

from math import inf

def cal_max_sum(arr):

    max_so_far = -inf
    max_ending_here = 0
    start = 0
    end = 0
    s = 0

    for i in range(len(arr)):
        max_ending_here = max_ending_here + arr[i]

        if max_so_far < max_ending_here:
            max_so_far = max_ending_here
            start = s
            end = i
    
        if max_ending_here < 0:
            max_ending_here = 0
            s = i+1

    print ("Maximum contiguous sum is %d" %(max_so_far))
    print("Starting Index %d" % (start))
    print("Ending Index %d" % (end))

if __name__ == "__main__":
    a = [-2, -3, -4, -1, -2, -1, -5, -3]
    cal_max_sum(a)
