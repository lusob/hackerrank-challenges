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

# https://www.hackerrank.com/challenges/weighted-uniform-string
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from collections import Counter
if __name__=='__main__':
    s = raw_input().strip()
    n = int(raw_input().strip())
    kk  = Counter(s)
    d={}
    aux = 1
    for ix,i in enumerate(s):
        if ix<len(s)-1 and s[ix+1] == i:
            aux += 1
        else:
            if i not in d.keys() or d[i]<aux:
                d[i] = aux
            aux = 1

    def has_weight(x):
        m = dict(map(lambda (k,v):(ord(k)-ord('a')+1,v), d.iteritems()))
        for k,v in m.iteritems():
            if x%k == 0 and x/k <= v:
                return True
        return False

    for a0 in xrange(n):
        x = int(raw_input().strip())
        print  'Yes' if has_weight(x) else 'No'

# https://www.hackerrank.com/challenges/separate-the-numbers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__=='__main__':
    def is_beutiful_string(s):
        if s[0] == '0':
            return None
        else:
            first = None
            for i in range(1,len(s)/2+1):
                first = s[:i]
                k = first
                j = 0
                while j < len(s)-len(str(k)):
                    n_chars = len(str(k))
                    p = s[j:j+n_chars]
                    k = int(p) + 1
                    if j == 0:
                        first = s[:n_chars]
                    n = s[j+n_chars:j+n_chars+len(str(k))]
                    if k != int(n):
                        first = None
                        break
                    j += n_chars
                if first != None:
                    return first
            return first

    q = int(raw_input().strip())
    for a0 in xrange(q):
        s = raw_input().strip()
        r = is_beutiful_string(s)
        print 'YES ' + r  if r else 'NO'


# https://www.hackerrank.com/challenges/drawing-book
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__=='__main__':
	def solve(n, p):
		return min(n/2-p/2, p/2)
	n = int(raw_input().strip())
	p = int(raw_input().strip())
	result = solve(n, p)
	print(result)

# https://www.hackerrank.com/challenges/counting-valleys
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n = input()
s = raw_input()
valleys = 0
sea_level_offset = 0
for ix, i in enumerate(s):
    sea_level_offset += 1 if i == 'U' else -1
    #if i == 'U' and s[ix+sea_level_offset:ix] == sea_level_offset*'D':
    if sea_level_offset == 0 and s[ix]=='U':
        valleys += 1
print valleys

# https://www.hackerrank.com/challenges/electronics-shop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def getMoneySpent(keyboards, drives, s):
    l = [k+d if k+d <= s else 0 for k in keyboards if k<s for d in drives if d<s]
    return max(l) if len(l) else -1

s,n,m = raw_input().strip().split(' ')
s,n,m = [int(s),int(n),int(m)]
keyboards = map(int, raw_input().strip().split(' '))
drives = map(int, raw_input().strip().split(' '))
#  The maximum amount of money she can spend on a keyboard and USB drive, or -1 if she can't purchase both items
moneySpent = getMoneySpent(keyboards, drives, s)
print(moneySpent)

# https://www.hackerrank.com/challenges/magic-square-forming/problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
diffs = []
s = []
for s_i in range(3):
	s_t = [int(s_temp) for s_temp in raw_input().strip().split(' ')]
	s+=s_t

all_magic_squares= [
	[8, 1, 6, 3, 5, 7, 4, 9, 2],
	[6, 1, 8, 7, 5, 3, 2, 9, 4],
	[4, 9, 2, 3, 5, 7, 8, 1, 6],
	[2, 9, 4, 7, 5, 3, 6, 1, 8],
	[8, 3, 4, 1, 5, 9, 6, 7, 2],
	[4, 3, 8, 9, 5, 1, 2, 7, 6],
	[6, 7, 2, 1, 5, 9, 8, 3, 4],
	[2, 7, 6, 9, 5, 1, 4, 3, 8]]

#compare s to each in all possible get number of differences for each to diffs
for p in all_magic_squares:
	cost = 0
	for p_i, s_i in list(zip(p,s)):
		if p_i != s_i:
			cost += abs(p_i - s_i)
	diffs.append(cost)
print(min(diffs))

# https://www.hackerrank.com/challenges/picking-numbers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from collections import Counter
n = int(raw_input().strip())
a = map(int,raw_input().strip().split(' '))
cc = Counter(a)
print max([cc[k-1]+cc[k] for k in cc.keys()])

# https://www.hackerrank.com/challenges/climbing-the-leaderboard
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n = int(raw_input().strip())
scores = map(int,raw_input().strip().split(' '))
m = int(raw_input().strip())
alice = map(int,raw_input().strip().split(' '))
leader_board = sorted(set(scores))
ix = 0
for current_score in alice:
    while ix<len(leader_board) and current_score>=leader_board[ix]:
        ix+=1
    pos = max(1,len(leader_board)+1-ix)
    print pos

# https://www.hackerrank.com/challenges/designer-pdf-viewer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
h = map(int, raw_input().strip().split(' '))
word = raw_input().strip()
ord_a = ord('a')
print len(word)*max([h[ord(c)-ord_a] for c in word])

# https://www.hackerrank.com/challenges/beautiful-days-at-the-movies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
i, j, k = map(int, raw_input().strip().split(' '))
beautiful_days = 0
for n in range(i,j):
	if abs(n-int(str(n)[::-1])) % k == 0:
		beautiful_days += 1
print beautiful_days

# https://www.hackerrank.com/challenges/strange-advertising/problem
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n = input()-1
m = 5
likes = 2
total_likes = 2
for i in range(n):
	m = likes * 3
	likes = m//2
	total_likes += likes
print total_likes

