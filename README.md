# leetcode_sol

All the solution is written by python3.

[toc]
## Array
## String
### 2437. Number of Valid Clock Times
You are given a string of length `5` called time, representing the current time on a digital clock in the format `"hh:mm"`. The earliest possible time is `"00:00"` and the latest possible time is `"23:59"`.
In the string `time`, the digits represented by the `?` symbol are **unknown**, and must be **replaced** with a digit from `0` to `9`.

Return an integer `answer`, the number of valid clock times that can be created by replacing every `?` with a digit from `0` to `9`.

sol: find all the situation that the time is valid.

```python
class Solution:
    def countTime(self, time: str) -> int:
        hour = 0
        minute = 0

        '''find the number of valid hour'''
        for i in range(24):
            if (time[0] == "?" or i // 10 == int(time[0])) and (time[1] == "?" or i % 10 == int(time[1])):
                hour += 1

        '''find the number of valid minute'''
        for i in range(60):
            if (time[3] == "?" or i // 10 == int(time[3])) and (time[4] == "?" or i % 10 == int(time[4])):
                minute += 1

        return hour * minute
```
## Linked List
## Stack
## Queue
## Tree
## Hash Table
## Two Pointers
## Dynamic Programming
## Backtracking
## Greedy
## Breadth-first Search
## Depth-first Search