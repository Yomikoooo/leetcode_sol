# leetcode_sol

All the solution is written by python3.

- [leetcode\_sol](#leetcode_sol)
  - [Array](#array)
  - [String](#string)
    - [1016. (medium) Binary String With Substrings Representing 1 To N](#1016-medium-binary-string-with-substrings-representing-1-to-n)
    - [2437. (easy) Number of Valid Clock Times](#2437-easy-number-of-valid-clock-times)
  - [Linked List](#linked-list)
    - [19.(Medium) Remove Nth Node From End of List](#19medium-remove-nth-node-from-end-of-list)
    - [24.(Medium) Swap Nodes in Pairs](#24medium-swap-nodes-in-pairs)
  - [Stack](#stack)
  - [Queue](#queue)
  - [Tree](#tree)
  - [Hash Table](#hash-table)
  - [Two Pointers](#two-pointers)
  - [Dynamic Programming](#dynamic-programming)
  - [Backtracking](#backtracking)
  - [Greedy](#greedy)
  - [Breadth-first Search](#breadth-first-search)
  - [Depth-first Search](#depth-first-search)

## Array
## String

### 1016. (medium) Binary String With Substrings Representing 1 To N
Given a binary string `s` and a positive integer `n`, return true if the binary representation of all the integers in the range `[1, n]` are substrings of `s`, or false otherwise.

A substring is a contiguous sequence of characters within a string.

**sol**: brute force.
```python
class Solution:
    def queryString(self, s: str, n: int) -> bool:
        return all(bin(i)[2:] in s for i in range(1, n + 1))
```
**notes**: `bin(i)[2:]` return the binary representation of `i` without prefix `0b`. `all()` return `True` if all elements in the iterable are true, otherwise return `False`.</br>


### 2437. (easy) Number of Valid Clock Times
You are given a string of length `5` called time, representing the current time on a digital clock in the format `"hh:mm"`. The earliest possible time is `"00:00"` and the latest possible time is `"23:59"`.
In the string `time`, the digits represented by the `?` symbol are **unknown**, and must be **replaced** with a digit from `0` to `9`.

Return an integer `answer`, the number of valid clock times that can be created by replacing every `?` with a digit from `0` to `9`.

**sol**: brute force.

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
setting up a linked list class
```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```
### 19.(Medium) Remove Nth Node From End of List
Given the `head` of a linked list, remove the `nth` node from the end of the list and return its head.
```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        res = ListNode(next=head)
        fast, slow = res, res

        while n>=0: # create a window and length equals n
            fast = fast.next
            n -= 1
        while (fast!=None): # let the fast pointer go to the end
            slow = slow.next
            fast = fast.next
        
        slow.next = slow.next.next # remove
        return res.next

```

### 24.(Medium) Swap Nodes in Pairs
Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)
```python
class Solution:
    ### example: 1->2->3->4
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        res = ListNode(next=head) #dummy node
        prev = res # pointer
        while prev.next and prev.next.next: #must have next 2 nodes
            cur=prev.next #first node
            post= prev.next.next #second node

            #swap cur=1, post=2 to cur=2, post=1
            cur.next=post.next # 1st node's next 3rd node
            post.next=cur # 2nd node's next 1st node
            prev.next=post # dummy node's next 2nd node(1st node after swap)

            prev=prev.next.next # current pointer move to 3rd node
        return res.next #return head
```
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