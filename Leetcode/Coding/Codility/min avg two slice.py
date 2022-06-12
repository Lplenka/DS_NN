#https://app.codility.com/programmers/lessons/5-prefix_sums/min_avg_two_slice/

def solution(A):
    # write your code in Python 3.6
    n = len(A)
    psum = [0]
    for k in range(0, n):
        psum.append(psum[k] + A[k])

    lft_idx = 0
    min_idx = 0

    curr_avg = (A[0] + A[1])/2
    min_avg = (A[0] + A[1])/2

    for i in range(2, n):
        avg_prev = (psum[i+1] - psum[lft_idx])/(i-lft_idx+1)
        # print("avg_prev ",avg_prev)

        avg_of_two = (A[i-1] + A[i])/2

        # print("avg_of_two ",avg_of_two)

        if(avg_of_two<avg_prev):
            curr_avg = avg_of_two
            lft_idx = i-1
        else:
            curr_avg = avg_prev

        # print("curr_avg ",curr_avg)
        # print("min_idx ",min_idx)

        if(curr_avg<min_avg):
            min_avg=curr_avg
            min_idx = lft_idx

    return min_idx