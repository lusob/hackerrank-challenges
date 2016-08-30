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

