#https://codility.com/media/train/3-PrefixSums.pdf
def prefix_sums(A):
    n = len(A)
    psum = [0]
    for k in range(0, n):
        psum.append(psum[k] + A[k])
    return psum

def suffix_sum(P, x, y):
    return P[y+1] - P[x]


def mushrooms(A, k, m):
    n = len(A)
    result = 0
    pref_sum = prefix_sums(A)
    for p in range(min(k,m)):
        left_pos = k-p
        right_pos = min(n-1, max(k, k + m-2*p))
        result = max(result, suffix_sum(pref_sum, left_pos, right_pos))


    for p in range(min(n-k, m)):
        right_pos = k+p
        left_pos = max(0, min(k, k - (m-2*p)))
        result = max(result, suffix_sum(pref_sum, left_pos, right_pos))
    
    return result

if __name__ == '__main__':
    l = [2, 3, 7, 5, 1, 3, 9]
    print(mushrooms(l, 4, 6))

