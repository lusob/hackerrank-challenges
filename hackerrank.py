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

