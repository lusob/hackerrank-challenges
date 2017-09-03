# My solutions to hackerrank challenges,
# paste every snippet on hackerrank web to solve it

# https://www.hackerrank.com/challenges/mark-and-toys
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Enter your code here. Read input from STDIN. Print output to STDOUT
def max_toys(prices, rupees):
  #Compute and return final answer over here
  spent = []
  prices.sort()
  for toy_price in prices:
    if rupees > toy_price:
        rupees -= toy_price
        spent.append(toy_price)
    else:
        break;
  return len(spent)

if __name__ == '__main__':
  n, k = map(int, raw_input().split())
  prices = map(int, raw_input().split())
  print max_toys(prices, k)

# https://www.hackerrank.com/challenges/two-arrays
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def exist_permutation():
    for i in xrange(n):
        if A[i]+B[i]<k:
            return 'NO'
    return 'YES'

if __name__ == '__main__':
    T = input()
    for i in xrange(T):
        n, k = map(int, raw_input().split())
        A = map(int, raw_input().split())
        B = map(int, raw_input().split())
        A.sort()
        B.sort(reverse=True)
        print exist_permutation()

# https://www.hackerrank.com/challenges/sherlock-and-array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def is_valid(N, A):
    if N == 1:
        return 'YES'
    sum_left = 0
    sum_right = sum(A) - A[0]
    for i in xrange(1,N-1):
        sum_left += A[i-1]
        sum_right -= A[i]
        if sum_left == sum_right:
            return 'YES'
    return 'NO'

if __name__ == '__main__':
    T = input()
    for case in xrange(T):
        N = int(raw_input())
        A = map(int, raw_input().split())
        print is_valid(N, A)

# https://www.hackerrank.com/challenges/missing-numbers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    from collections import Counter
    N = input()
    A = Counter(map(int, raw_input().split()))
    M = input()
    B = Counter(map(int, raw_input().split()))
    print ' '.join(sorted(map(str,(B-A).keys())))

# https://www.hackerrank.com/challenges/pairs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def pairs(n,k):
    c = 0
    for i in xrange(N):
        for j in xrange(i+1,N):
            diff = abs(n[i] - n[j])
            if diff == k:
                c += 1
                break
            elif diff > k:
                break
    return c
if __name__ == '__main__':
    N, K = map(int, raw_input().split())
    numbers = map(int, raw_input().split())
    numbers.sort()
    print pairs(numbers, K)

# https://www.hackerrank.com/challenges/connected-cell-in-a-grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def region_sum(i, j):
    sum = 0
    if board[i][j]:
        sum = 1
        board[i][j] = 0
        if j<m-1:
            # right cell
            sum += region_sum(i,j+1)
            # right diagonal cells
            if i>0:
                sum += region_sum(i-1,j+1)
            if i<n-1:
                sum += region_sum(i+1,j+1)
        if j>0:
            # left cell
            sum += region_sum(i,j-1)
            # left diagonal cells
            if i>0:
                sum += region_sum(i-1,j-1)
            if i<n-1:
                sum += region_sum(i+1,j-1)
        # top cell
        if i>0:
            sum += region_sum(i-1,j)
        # bottom cell
        if i<n-1:
            sum += region_sum(i+1,j)
    return sum

if __name__ == '__main__':
    n = input()
    m = input()
    board = []
    max_sum = 0
    for i in xrange(n):
        board.append(map(int, raw_input().split()))
    for i in xrange(n):
        for j in xrange(m):
            if board[i][j]:
                rg_sum = region_sum(i,j)
                max_sum = max(max_sum, rg_sum)
    print max_sum

# https://www.hackerrank.com/challenges/equal-stacks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n1, n2, n3 = map(int, raw_input().split())
s1 = map(int, raw_input().split())
s2 = map(int, raw_input().split())
s3 = map(int, raw_input().split())
s1.reverse()
s2.reverse()
s3.reverse()
s1_sum = s1[0]
s2_sum = s2[0]
s3_sum = s3[0]
res = s1_sum if s1_sum == s2_sum == s3_sum else 0
i = j = k = 1
while i < n1 and j < n2 and k < n3:
    s1_sum += s1[i]
    i += 1
    max_sum = max(s1_sum, s2_sum, s3_sum)
    while j < n2 and s2_sum + s2[j] <= max_sum:
        s2_sum += s2[j]
        j += 1
        while k < n3 and s3_sum + s3[k] <= max_sum:
            s3_sum += s3[k]
            k += 1
    res = s1_sum if s1_sum == s2_sum == s3_sum else res
print res

# https://www.hackerrank.com/contests/w23/challenges/gears-of-war
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
q = input()
if __name__ == '__main__':
    for i in xrange(q):
        n = input()
        print 'Yes' if n%2 == 0 else 'No'

# https://www.hackerrank.com/contests/w23/challenges/lighthouse
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_circle_rad(ci, cj):
    for r in xrange(1, n):
        for offset_i in xrange(-r,r+1):
            for offset_j in xrange(-r,r+1):
                i = ci+offset_i
                j = cj+offset_j
                # if it's inside the euclidean space but it's out ot grid or is not a dot
                if((offset_i**2+offset_j**2)<= r**2 and ((i<0 or j<0 or i>=n or j>=n) or N[i][j] != '.')):
                    return r-1

if __name__ == '__main__':
    n = input()
    N = []
    max_rad = 0
    for i in xrange(n):
        N.append(raw_input())
    for i in xrange(n):
        for j in xrange(n):
            max_rad = max(max_rad, get_circle_rad(i,j))
    print max_rad

# https://www.hackerrank.com/contests/w23/challenges/treasure-hunting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def orto(a) :
    b=[]
    b.append(-a[1])
    b.append(a[0])
    return b

if __name__ == '__main__':
    x, y = map(int,raw_input().split())
    a, b = map(int,raw_input().split())
    a_, b_ = orto([a, b])
    n  = float(a*y-x*b)/float(b_*a-a_*b)
    k = float(x-n*a_) / float(a)
    print k
    print n

# https://www.hackerrank.com/contests/w23/challenges/commuting-strings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def subpattern(s):
    import re
    for i in xrange(1,len(s)/2):
        if len(s)%len(s[0:i]) == 0 and len(re.findall(s[0:i], s)) == len(s)/len(s[0:i]):
            return (m/len(s[0:i])) % (10**9+7)
    return m/len(s)
if __name__ == '__main__':
    s = raw_input()
    m = input()
    print subpattern(s)

# https://www.hackerrank.com/challenges/candies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this solution isn't passing all test cases
#
if __name__ == '__main__':
    n = input()
    ratings = []
    total_candies_given = 1
    last_candies_given = 1
    for i in xrange(n):
        ratings.append(input())
    for i in xrange(n-1):
        current = i
        next = i+1
        if ratings[current] >= ratings[next]:
            last_candies_given = 1
            while next < n and ratings[current] > ratings[next]:
                last_candies_given += 1
                current += 1
                next +=1
            total_candies_given += last_candies_given
            last_candies_given = 0
        else:
            last_candies_given += 1
        total_candies_given += last_candies_given
    print total_candies_given

# https://www.hackerrank.com/challenges/longest-increasing-subsequent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__=='__main__':
    n = input()
    ratings = []
    for i in xrange(n):
        ratings.append(input())

    def my_longest_increasing_subsequence(d):
        l = []
        for i in xrange(len(d)):
            seqs = [l[j] for j in xrange(i) if d[i] > l[j][-1]] or [[]]
            l.append(max(seqs, key=len) + [d[i]])
        return max(l, key=len)

    print len(my_longest_increasing_subsequence(ratings))

# https://www.hackerrank.com/challenges/dynamic-programming-classics-the-longest-common-subsequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__=='__main__':
    m, n = map(int, raw_input().split())
    A = map(int, raw_input().split())
    B = map(int, raw_input().split())
    C = [[0]*(n+1) for j in range(m+1)]
    def LCSLength(X, Y, m, n):
        for i in xrange(m):
            for j in xrange(n):
                if X[i] == Y[j]:
                    C[i][j] = C[i-1][j-1] + 1
                else:
                    C[i][j] = max(C[i][j-1], C[i-1][j])
        return C[m-1][n-1]

    def backtrack(C, X, Y, i, j):
        if i == -1 or j == -1:
            return ""
        elif  X[i] == Y[j]:
            return "%s %s" % (backtrack(C, X, Y, i-1, j-1), X[i])
        else:
            if C[i][j-1] > C[i-1][j]:
                return backtrack(C, X, Y, i, j-1)
            else:
                return backtrack(C, X, Y, i-1, j)

    LCSLength(A, B, m, n)
    print backtrack(C, A, B, m-1, n-1)

# https://www.hackerrank.com/challenges/two-characters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def is_valid_t(s):
    return all(i!=s[x+1] for x,i in enumerate(s[:-1]))

if __name__=='__main__':
    s_len = int(raw_input().strip())
    s = raw_input().strip()
    ss = set(s)
    len_t = 0
    for x,i in enumerate(ss):
        for j in ss.difference(set([i])):
            t = [k for k in s if k==i or k==j]
            if is_valid_t(t):
                len_t = max(len_t, len(t))
    print len_t

# https://www.hackerrank.com/challenges/mars-exploration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__=='__main__':
    S = raw_input().strip()
    print sum([int(S[x]!='S') + int(S[x+1]!='O') + int(S[x+2]!='S') for x in range(0, len(S), 3)])

# https://www.hackerrank.com/challenges/hackerrank-in-a-string
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__=='__main__':
    q = int(raw_input().strip())
    for a0 in xrange(q):
        s = raw_input().strip()
        t = list('hackerrank')
        c = t.pop(0)
        for i in s:
            if len(t) == 0:
                break
            if i == c:
                c = t.pop(0)
        print 'NO' if len(t) else 'YES'

