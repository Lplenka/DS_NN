def sieve_func(n):
    sieve = [1]*(n+1)
    i = 2
    while(i*i<n):
        if(sieve[i]):
            for j in range(i*i, n+1, i):
                sieve[j] = 0
        i+=1

    for p in range(2, n+1):
        if sieve[p]:
            print(p)

if __name__ == '__main__':
    sieve_func(17)