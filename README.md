# leetcode_sol

All the solution is written by python3.

- [leetcode\_sol](#leetcode_sol)
  - [Array](#array)
    - [26. (easy) Remove Duplicates from Sorted Array](#26-easy-remove-duplicates-from-sorted-array)
    - [283. (easy) Move Zeroes](#283-easy-move-zeroes)
    - [977. (easy) Squares of a Sorted Array](#977-easy-squares-of-a-sorted-array)
    - [209. (medium) Minimum Size Subarray Sum](#209-medium-minimum-size-subarray-sum)
    - [904. (medium) Fruit Into Baskets](#904-medium-fruit-into-baskets)
    - [54. (medium) Spiral Matrix](#54-medium-spiral-matrix)
    - [59. (medium) Spiral Matrix II](#59-medium-spiral-matrix-ii)
    - [76. (hard) Minimum Window Substring](#76-hard-minimum-window-substring)
  - [String](#string)
    - [1016. (medium) Binary String With Substrings Representing 1 To N](#1016-medium-binary-string-with-substrings-representing-1-to-n)
    - [1419. (medium) Minimum Number of Frogs Croaking](#1419-medium-minimum-number-of-frogs-croaking)
    - [2432. (easy) The Employee That Worked on the Longest Task](#2432-easy-the-employee-that-worked-on-the-longest-task)
    - [2437. (easy) Number of Valid Clock Times](#2437-easy-number-of-valid-clock-times)
  - [Linked List](#linked-list)
    - [203. (easy) Remove Linked List Elements](#203-easy-remove-linked-list-elements)
    - [206. (easy) Reverse Linked List](#206-easy-reverse-linked-list)
    - [19.(medium) Remove Nth Node From End of List](#19medium-remove-nth-node-from-end-of-list)
    - [24.(medium) Swap Nodes in Pairs](#24medium-swap-nodes-in-pairs)
    - [142. (medium) Linked List Cycle II](#142-medium-linked-list-cycle-ii)
    - [160. (easy?) Intersection of Two Linked Lists](#160-easy-intersection-of-two-linked-lists)
  - [Stack](#stack)
    - [844. (easy) Backspace String Compare](#844-easy-backspace-string-compare)
  - [Queue](#queue)
  - [Tree](#tree)
  - [Hash Table](#hash-table)
  - [Dynamic Programming](#dynamic-programming)
  - [Backtracking](#backtracking)
  - [Greedy](#greedy)
  - [Breadth-first Search](#breadth-first-search)
  - [Depth-first Search](#depth-first-search)

## Array

### 26. (easy) Remove Duplicates from Sorted Array
Given an integer array `nums` sorted in **non-decreasing order**, remove the duplicates **in-place** such that each unique element appears only **once**. The **relative order** of the elements should be kept the **same**.
```
Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
```
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        n = len(nums)
        j=0
        for i in range(n):
            if nums[i]!= nums[j]:
                j+=1
                nums[j]=nums[i]
        return j+1
```
### 283. (easy) Move Zeroes
Given an integer array `nums`, move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.
```
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]

Input: nums = [0]
Output: [0]
```
```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        j=0
        n= len(nums)
        for i in range(n):
            if nums[i]!=0:
                nums[j],nums[i]=nums[i],nums[j]
                j+=1
        return nums
```
### 977. (easy) Squares of a Sorted Array
Given an integer array `nums` sorted in **non-decreasing** order, return *an array of **the squares of each number** sorted in non-decreasing order*.
```
Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].

Input: nums = [-7,-3,2,3,11]
Output: [4,9,9,49,121]
```
```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        n = len(nums)
        i=0
        j=n-1
        ans = []
        while i<=j:
            if abs(nums[i])<abs(nums[j]):
                ans.append(nums[j]**2)
                j-=1
            else:
                ans.append(nums[i]**2)
                i+=1
        return ans[::-1]
```
### 209. (medium) Minimum Size Subarray Sum
Given an array of positive integers `nums` and a positive integer `target`, return the minimal length of a **contiguous subarray** `[numsl, numsl+1, ..., numsr-1, numsr]` of which the sum is greater than or equal to `target`. If there is no such subarray, return `0` instead.
```
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.

Input: target = 4, nums = [1,4,4]
Output: 1
```
```python
# moving window
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        ans = n+1
        i=0
        j=0
        cur = 0
        while i < n:
            cur +=nums[i]
            while cur >= target:
                ans = min(ans, i-j+1)
                cur -=nums[j]
                j+=1
            i+=1
        return 0 if ans == n+1 else ans 
```
### 904. (medium) Fruit Into Baskets
You are visiting a farm that has a single row of fruit trees arranged from left to right. The trees are represented by an integer array `fruits` where `fruits[i]` is the type of fruit the `ith` tree produces.
You want to collect as much fruit as possible. However, the owner has some strict rules that you must follow:
- You only have **two** baskets, and each basket can only hold a **single type** of fruit. There is no limit on the amount of fruit each basket can hold.
- Starting from any tree of your choice, you must pick **exactly one fruit** from **every** tree (including the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
- Once you reach a tree with fruit that cannot fit in your baskets, you must stop.
Given the integer array `fruits`, return *the **maximum** number of fruits you can pick*.
```
Input: fruits = [1,2,1]
Output: 3
Explanation: We can pick from all 3 trees.

Input: fruits = [0,1,2,2]
Output: 3
Explanation: We can pick from trees [1,2,2].
If we had started at the first tree, we would only pick from trees [0,1].

Input: fruits = [1,2,3,2,2]
Output: 4
Explanation: We can pick from trees [2,3,2,2].
If we had started at the first tree, we would only pick from trees [1,2].
```
```python
# moving window
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        j=0
        n = len(fruits)
        total=0
        ans =0
        cat=defaultdict(int)
        for i in range(n): # move the right pointer
            cat[fruits[i]]+=1
            if cat[fruits[i]]==1:
                total+=1 # total number of kinds
            while total>2: # no more than 2 kinds
                cat[fruits[j]]-=1 # move the left pointer
                if cat[fruits[j]]==0:
                    total -=1
                j+=1
            ans = max(ans, i-j+1) # update the max length
        return ans
```
### 54. (medium) Spiral Matrix
Given an `m x n` `matrix`, return *all elements of the* `matrix` *in spiral order*.
```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        l,r,t,b= 0, len(matrix[0])-1 , 0, len(matrix)-1 # left, right, top, bottom
        num = 1
        tar= len(matrix[0])*len(matrix) # matrix size
        ans=[]
        while len(ans) < tar:
            # left to right
            for i in range(l, r + 1):
                if len(ans)< tar:
                    ans.append(matrix[t][i])
                    num += 1
            t += 1
            # top to bottom
            for i in range(t, b + 1):
                if len(ans)< tar:
                    ans.append(matrix[i][r])
                    num += 1
            r -= 1
            # right to left
            for i in range(r, l - 1, -1):
                if len(ans)< tar:
                    ans.append(matrix[b][i])
                    num += 1
            b -= 1
            # bottom to top
            for i in range(b, t - 1, -1):
                if len(ans)< tar:
                    ans.append(matrix[i][l])
                    num += 1
            l += 1
        return ans
```
### 59. (medium) Spiral Matrix II
Given a positive integer `n`, generate an `n x n` `matrix` filled with elements from `1` to `n2` in spiral order.
```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        l, r, t, b = 0, n - 1, 0, n - 1
        mat = [[0 for _ in range(n)] for _ in range(n)]
        num, tar = 1, n * n
        while num <= tar:
            for i in range(l, r + 1): # left to right
                mat[t][i] = num
                num += 1
            t += 1
            for i in range(t, b + 1): # top to bottom
                mat[i][r] = num
                num += 1
            r -= 1
            for i in range(r, l - 1, -1): # right to left
                mat[b][i] = num
                num += 1
            b -= 1
            for i in range(b, t - 1, -1): # bottom to top
                mat[i][l] = num
                num += 1
            l += 1
        return mat

```
### 76. (hard) Minimum Window Substring
Given two strings `s` and `t` of lengths `m` and `n` respectively, return the **minimum window substring** of `s` such that every character in `t` (**including duplicates**) is included in the window. If there is no such substring, return the empty string `""`.
```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.

Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.
```
```python
# moving window
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        j = 0
        # count each char in t
        cat = collections.defaultdict(int)
        res=(0,float('inf'))
        for c in t:
            cat[c]+=1
        needcat=len(t)

        # moving window
        for i,c in enumerate(s):
            if cat[c]>0:
                needcat-=1
            cat[c]-=1
            if needcat==0: # have involved all chars in t
                while True:
                    # if not lose any char in t, slow pointer move right
                    c=s[j]
                    if cat[c]==0:
                        break
                    cat[c]+=1 #
                    j+=1
                # update smallest length and move fast pointer right until next window
                if i-j<res[1]-res[0]:
                    res=(j,i)
                cat[s[j]]+=1
                needcat+=1
                j+=1
        return '' if res[1]>len(s) else s[res[0]:res[1]+1]
```
## String

### 1016. (medium) Binary String With Substrings Representing 1 To N
Given a binary string `s` and a positive integer `n`, return true if the binary representation of all the integers in the range `[1, n]` are substrings of `s`, or false otherwise.

A substring is a contiguous sequence of characters within a string.

```python
#brute force
class Solution:
    def queryString(self, s: str, n: int) -> bool:
        return all(bin(i)[2:] in s for i in range(1, n + 1))
```
**notes**: `bin(i)[2:]` return the binary representation of `i` without prefix `0b`. `all()` return `True` if all elements in the iterable are true, otherwise return `False`.</br>
### 1419. (medium) Minimum Number of Frogs Croaking
Given the string `croakOfFrogs`, which represents a combination of the string "croak" from different frogs, that is, multiple frogs can croak at the same time, so multiple “croak” are mixed. Return the minimum number of different frogs to finish all the croak in the given string.
```python
class Solution:
    def minNumberOfFrogs(self, croakOfFrogs: str) -> int:
        # remove the cases no need to consider
        if len(croakOfFrogs) % 5 != 0: 
            return -1
        
        # create a dict
        idx = {c: i for i, c in enumerate('croak')} # {'c': 0, 'r': 1, 'o': 2, 'a': 3, 'k': 4}
        cnt = [0] * 5 # [0, 0, 0, 0, 0]
        ans = x = 0
        # 
        for i in map(idx.get, croakOfFrogs): # return the value of the key in the dict
            cnt[i] += 1 # count the number of each letter
            if i == 0:  # count frogs(new frog croak)
                x += 1  # x is the number of frogs are croaking
                ans = max(ans, x) # update
            else: 
                if cnt[i - 1] == 0: # if not consecutive, invalid
                    return -1
                cnt[i - 1] -= 1 # croaked number
                if i == 4: # a frog finish croaking
                    x -= 1
        return -1 if x else ans
```
```
Input: croakOfFrogs = "croakcroak"
Output: 1 
Explanation: One frog yelling "croak" twice.

Input: croakOfFrogs = "crcoakroak"
Output: 2 
Explanation: The minimum number of frogs is two. 
The first frog could yell "crcoakroak".
The second frog could yell later "crcoakroak".

Input: croakOfFrogs = "croakcrook"
Output: -1
Explanation: The given string is an invalid combination of "croak" from different frogs.
```

### 2432. (easy) The Employee That Worked on the Longest Task
There are `n` employees, each with a unique id from `0` to `n - 1`.

You are given a 2D integer array `logs` where `logs[i] = [idi, leaveTimei]` where:

`idi` is the id of the employee that worked on the `ith` task, and
`leaveTimei` is the time at which the employee finished the `ith` task. All the values `leaveTimei` are unique.
Note that the `ith` task starts the moment right after the `(i - 1)th` task ends, and the `0th` task starts at time `0`.

Return the id of the employee that worked the task with the longest time. If there is a tie between two or more employees, return *the* ***smallest*** *id among them*.
```python
class Solution:
    def hardestWorker(self, n: int, logs: List[List[int]]) -> int:
        l = len(logs)
        mx = 0
        cur = 0

        for i in range(l):
            if i==0: # border case
                mx = logs[0][1]
                cur= logs[0][0]
            if mx < logs[i][1]-logs[i-1][1]: # update the max time and the id of the employee
                mx = max(mx, logs[i][1]-logs[i-1][1])
                cur = logs[i][0]
            if mx == logs[i][1]-logs[i-1][1]: # if there is a tie, return the smallest id
                cur = min(cur, logs[i][0])
        return cur
```
```
Input: n = 10, logs = [[0,3],[2,5],[0,9],[1,15]]
Output: 1
Explanation: 
Task 0 started at 0 and ended at 3 with 3 units of times.
Task 1 started at 3 and ended at 5 with 2 units of times.
Task 2 started at 5 and ended at 9 with 4 units of times.
Task 3 started at 9 and ended at 15 with 6 units of times.
The task with the longest time is task 3 and the employee with id 1 is the one that worked on it, so we return 1.
```

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

# setting up a linked list
class Node:
    def __init__(self, _val):
        self.val = _val
        self.prev = None
        self.next = None


class MyLinkedList:

    def __init__(self):
        self.head = Node(-1)
        self.tail = Node(-1)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def getNode(self, index: int) -> Node:
        isLeft = index < self.size / 2
        if not isLeft:
            index = self.size - index - 1
        cur = self.head.next if isLeft else self.tail.prev
        while cur != self.head and cur != self.tail:
            if index == 0:
                return cur
            index -= 1
            cur = cur.next if isLeft else cur.prev
        return None

    def get(self, index: int) -> int:
        node = self.getNode(index)
        return node.val if node else -1

    def addAtHead(self, val: int) -> None:
        node = Node(val)
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
        self.size += 1

    def addAtTail(self, val: int) -> None:
        node = Node(val)
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node
        self.size += 1


    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.size:
            return 
        if index <= 0:
            self.addAtHead(val)
        elif index == self.size:
            self.addAtTail(val)
        else:
            node, cur = Node(val), self.getNode(index)
            node.next = cur
            node.prev = cur.prev
            cur.prev.next = node
            cur.prev = node
            self.size += 1


    def deleteAtIndex(self, index: int) -> None:
        node = self.getNode(index)
        if node:
            node.prev.next = node.next
            node.next.prev = node.prev
            self.size -= 1



# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
```
### 203. (easy) Remove Linked List Elements
Given the `head` of a linked list and an integer `val`, remove all the nodes of the linked list that has `Node.val == val`, and return *the new head*.
```python
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy_head=ListNode(next=head)
        cur = dummy_head
        while cur.next != None:
            if cur.next.val == val:
                cur.next=cur.next.next
            else:
                cur = cur.next
        return dummy_head.next
```

### 206. (easy) Reverse Linked List
Given the `head` of a singly linked list, reverse the list, and return *the reversed list*.
```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr= None, head
        while curr is not None:
            next= curr.next
            curr.next=prev
            prev=curr
            curr=next
        return prev
```

### 19.(medium) Remove Nth Node From End of List
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

### 24.(medium) Swap Nodes in Pairs
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
### 142. (medium) Linked List Cycle II
Given the `head` of a linked list, return *the node where the cycle begins. If there is no cycle, return* `null`.
```python
def detectCycle(self, head):
        fast, slow = head, head
        while True:
            if not (fast and fast.next): return
            fast, slow = fast.next.next, slow.next #fast move 2 steps, slow move 1 step
            if fast == slow: break #if fast and slow meet, break
        fast = head #fast move to head, slow stay at the meeting point
        while fast != slow:
            fast, slow = fast.next, slow.next 
        return fast #return the new meeting point which is the entrance of the cycle
```

### 160. (easy?) Intersection of Two Linked Lists
Given the heads of two singly linked-lists `headA` and `headB`, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return `null`.
```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        A, B = headA, headB #Connect the tail of A to B, and the tail of B to A.
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A
```



## Stack

### 844. (easy) Backspace String Compare
Given two strings `s` and `t`, return `true` *if they are equal when both are typed into empty text editors. `'#'` means a backspace character*.
Note that after backspacing an empty text, the text will continue empty.
```
Input: s = "ab#c", t = "ad#c"
Output: true
Explanation: Both s and t become "ac".

Input: s = "ab##", t = "c#d#"
Output: true
Explanation: Both s and t become "".

Input: s = "a#c", t = "b"
Output: false
Explanation: s becomes "c" while t becomes "b".
```
```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        def build(s: str) -> str:
            ret = list()
            for i in s:
                if i != "#":
                    ret.append(i)
                elif ret:
                    ret.pop()
            return "".join(ret)
        
        return build(s) == build(t)

```
## Queue
## Tree
## Hash Table
## Dynamic Programming
## Backtracking
## Greedy
## Breadth-first Search
## Depth-first Search