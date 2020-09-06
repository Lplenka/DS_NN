# https: // www.geeksforgeeks.org/count-palindromic-subsequence-given-string/

"""
Pseudo Code
Initial Values: i = 0, j = n-1

CountPS(i, j)
// Every single character of a string is a palindrome
// subsequence
if i == j
return 1 // palindrome of length 1

// If first and last characters are same, then we
// consider it as palindrome subsequence and check
// for the rest subsequence(i+1, j), (i, j-1)
Else if (str[i] == str[j)]
return countPS(i+1, j) + countPS(i, j-1) + 1

else
// check for rest sub-sequence and remove common
// palindromic subsequences as they are counted
// twice when we do countPS(i+1, j) + countPS(i, j-1)
return countPS(i+1, j) + countPS(i, j-1) - countPS(i+1, j-1)
"""

# Python3 code to Count Palindromic
# Subsequence in a given String

# Function return the total
# palindromic subsequence


def countPS(str):

    N = len(str)

    # Create a 2D array to store the count
    # of palindromic subsequence
    cps = [[0 for i in range(N + 2)]for j in range(N + 2)]

    # palindromic subsequence of length 1
    for i in range(N):
        cps[i][i] = 1

    # check subsequence of length L
    # is palindrome or not
    for L in range(2, N + 1):

        for i in range(N):
            k = L + i - 1
            if (k < N):
                if (str[i] == str[k]):
                    cps[i][k] = (cps[i][k - 1] +
                                 cps[i + 1][k] + 1)
                else:
                    cps[i][k] = (cps[i][k - 1] +
                                 cps[i + 1][k] -
                                 cps[i + 1][k - 1])

    # return total palindromic subsequence
    return cps[0][N - 1]


# Driver program
str = "abcb"
print("Total palindromic subsequence are : ", countPS(str))
