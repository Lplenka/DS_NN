#https://www.geeksforgeeks.org/count-palindrome-sub-strings-string/

"""

Initial Values : i = 0, j = n-1;
Given string 'str'

CountPS(i, j)
   
   // If length of string is 2 then we 
   // check both character are same or not 
   If (j == i+1)
      return str[i] == str[j]

   Else If str[i..j] is PALINDROME 
      // increment count by 1 and check for 
      // rest palindromic substring (i, j-1), (i+1, j)
      // remove common palindrome substring (i+1, j-1)
      return  countPS(i+1, j) + countPS(i, j-1) + 1 -
                                   countPS(i+1, j-1);

    Else // if NOT PALINDROME 
       // We check for rest palindromic substrings (i, j-1)
       // and (i+1, j)
       // remove common palindrome substring (i+1 , j-1)
       return  countPS(i+1, j) + countPS(i, j-1) - 
                             countPS(i+1 , j-1);



"""
# Python 3 program to find palindromic
# substrings of a string

# Returns total number of palindrome
# substring of length greater then
# equal to 2


def CountPS(str, n):

    # creat empty 2-D matrix that counts
    # all palindrome substring. dp[i][j]
    # stores counts of palindromic
    # substrings in st[i..j]
    dp = [[0 for x in range(n)]
          for y in range(n)]

    # P[i][j] = true if substring str[i..j]
    # is palindrome, else false
    P = [[False for x in range(n)]
         for y in range(n)]

    # palindrome of single length
    for i in range(n):
        P[i][i] = True

    # palindrome of length 2
    for i in range(n - 1):
        if (str[i] == str[i + 1]):
            P[i][i + 1] = True
            dp[i][i + 1] = 1

    # Palindromes of length more than 2. This
    # loop is similar to Matrix Chain Multiplication.
    # We start with a gap of length 2 and fill DP
    # table in a way that the gap between starting and
    # ending indexes increase one by one by
    # outer loop.
    for gap in range(2, n):

        # Pick a starting point for the current gap
        for i in range(n - gap):

            # Set ending point
            j = gap + i

            # If current string is palindrome
            if (str[i] == str[j] and P[i + 1][j - 1]):
                P[i][j] = True

            # Add current palindrome substring ( + 1)
            # and rest palindrome substring (dp[i][j-1] +
            # dp[i+1][j]) remove common palindrome
            # substrings (- dp[i+1][j-1])
            if (P[i][j] == True):
                dp[i][j] = (dp[i][j - 1] +
                            dp[i + 1][j] + 1 - dp[i + 1][j - 1])
            else:
                dp[i][j] = (dp[i][j - 1] +
                            dp[i + 1][j] - dp[i + 1][j - 1])

    # return total palindromic substrings
    return dp[0][n - 1]


# Driver Code
if __name__ == "__main__":

    str = "abaab"
    n = len(str)
    print(CountPS(str, n))
