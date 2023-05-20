# leetcode_sol

All the solution is programmed by python3.
**types**: array, string, linked list, stack, queue, tree, hashmap, dynamic programming, backtracking, greedy

if **dynamic programming** with many images, attach the **link** and explain the idea and code, not the **description** of the problem.

- [leetcode\_sol](#leetcode_sol)
  - [Array](#array)
    - [26. (easy) Remove Duplicates from Sorted Array](#26-easy-remove-duplicates-from-sorted-array)
    - [27. (easy) Remove Element](#27-easy-remove-element)
    - [283. (easy) Move Zeroes](#283-easy-move-zeroes)
    - [704. (easy) Binary Search](#704-easy-binary-search)
    - [977. (easy) Squares of a Sorted Array](#977-easy-squares-of-a-sorted-array)
    - [1380. (easy) Lucky Numbers in a Matrix](#1380-easy-lucky-numbers-in-a-matrix)
    - [1886. (easy) Determine Whether Matrix Can Be Obtained By Rotation](#1886-easy-determine-whether-matrix-can-be-obtained-by-rotation)
    - [11. (medium) Container With Most Water](#11-medium-container-with-most-water)
    - [1985. (medium) Find the Kth Largest Integer in the Array](#1985-medium-find-the-kth-largest-integer-in-the-array)
    - [209. (medium) Minimum Size Subarray Sum](#209-medium-minimum-size-subarray-sum)
    - [904. (medium) Fruit Into Baskets](#904-medium-fruit-into-baskets)
    - [54. (medium) Spiral Matrix](#54-medium-spiral-matrix)
    - [59. (medium) Spiral Matrix II](#59-medium-spiral-matrix-ii)
    - [540. (medium) Single Element in a Sorted Array](#540-medium-single-element-in-a-sorted-array)
    - [76. (hard) Minimum Window Substring](#76-hard-minimum-window-substring)
  - [String](#string)
    - [13. (easy) Roman to Integer](#13-easy-roman-to-integer)
    - [2409. (easy) Count Days Spent Together](#2409-easy-count-days-spent-together)
    - [531. (medium) Lonely Pixel I](#531-medium-lonely-pixel-i)
    - [1003. (medium) Check If Word Is Valid After Substitutions](#1003-medium-check-if-word-is-valid-after-substitutions)
    - [1016. (medium) Binary String With Substrings Representing 1 To N](#1016-medium-binary-string-with-substrings-representing-1-to-n)
    - [1208. (medium) Get Equal Substrings Within Budget](#1208-medium-get-equal-substrings-within-budget)
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
    - [933. (easy) Number of Recent Calls](#933-easy-number-of-recent-calls)
  - [Tree](#tree)
    - [94. (easy) Binary Tree Inorder Traversal](#94-easy-binary-tree-inorder-traversal)
    - [965. (easy) Univalued Binary Tree](#965-easy-univalued-binary-tree)
    - [107. (medium) Binary Tree Level Order Traversal II](#107-medium-binary-tree-level-order-traversal-ii)
    - [1026. (medium) Maximum Difference Between Node and Ancestor](#1026-medium-maximum-difference-between-node-and-ancestor)
  - [Hashmap](#hashmap)
    - [1. (easy) Two Sum](#1-easy-two-sum)
    - [242. (easy) Valid Anagram](#242-easy-valid-anagram)
    - [383. (easy) Ransom Note](#383-easy-ransom-note)
    - [2441. (easy) Largest Positive Integer That Exists With Its Negative](#2441-easy-largest-positive-integer-that-exists-with-its-negative)
    - [49. (medium) Group Anagrams](#49-medium-group-anagrams)
    - [1072. (medium) Flip Columns For Maximum Number of Equal Rows](#1072-medium-flip-columns-for-maximum-number-of-equal-rows)
  - [Dynamic Programming](#dynamic-programming)
    - [413. (easy) Arithmetic Slices](#413-easy-arithmetic-slices)
    - [926. (medium) Flip String to Monotone Increasing](#926-medium-flip-string-to-monotone-increasing)
    - [1024. (medium) Video Stitching](#1024-medium-video-stitching)
    - [1027. (medium) Longest Arithmetic Subsequence](#1027-medium-longest-arithmetic-subsequence)
    - [1043. (medium) Partition Array for Maximum Sum](#1043-medium-partition-array-for-maximum-sum)
    - [1079. (medium) Letter Tile Possibilities](#1079-medium-letter-tile-possibilities)
    - [1105. (medium) Filling Bookcase Shelves](#1105-medium-filling-bookcase-shelves)
    - [1218. (medium) Longest Arithmetic Subsequence of Given Difference](#1218-medium-longest-arithmetic-subsequence-of-given-difference)
  - [Backtracking](#backtracking)
  - [Greedy](#greedy)
    - [1054. (medium) Distant Barcodes](#1054-medium-distant-barcodes)
    - [1909. (medium) Remove One Element to Make the Array Strictly Increasing](#1909-medium-remove-one-element-to-make-the-array-strictly-increasing)
  - [Simulation](#simulation)
    - [1073. (medium) Adding Two Negabinary Numbers](#1073-medium-adding-two-negabinary-numbers)
  - [Disjoint Set](#disjoint-set)
    - [1020. (medium) Number of Enclaves](#1020-medium-number-of-enclaves)
  - [Segment Tree](#segment-tree)

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
       '''
       for each element, if not equal to the previous one,
       then slow pointer move forward and change the value,
       to make sure the first j+1 elements are unique
       '''
        for i in range(n):
            if nums[i]!= nums[j]:
                j+=1
                nums[j]=nums[i]
        return j+1
```
### 27. (easy) Remove Element
Given an integer array `nums` and an integer `val`, remove all occurrences of `val` in `nums` **in-place**. The **relative order** of the elements may be changed.
```
Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 2.
It does not matter what you leave beyond the returned k (hence they are underscores).

Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,4,0,3,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums containing 0, 0, 1, 3, and 4.
Note that the five elements can be returned in any order.
It does not matter what you leave beyond the returned k (hence they are underscores).
```
```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        a=0
        b=0
        while a <len(nums):
            if nums[a]!=val:
                nums[b]= nums[a]
                b+=1
            a+=1
        return b
            
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
### 704. (easy) Binary Search
Given an array of integers `nums` which is sorted in ascending order, and an integer `target`, write a function to search `target` in `nums`. If `target` exists, then return its index. Otherwise, return `-1`.
```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
```
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l=0
        r=len(nums)-1
        while l<=r:
            mid = (l+r)//2
            if nums[mid]==target:
                return mid
            elif nums[mid]<target:
                l=mid+1
            else:
                r=mid-1
        return -1
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
### 1380. (easy) Lucky Numbers in a Matrix
Given a `m * n` matrix of **distinct** numbers, return all lucky numbers in the matrix in **any** order.
A lucky number is an element of the matrix such that it is the minimum element in its row and maximum in its column.
```
Input: matrix = [[3,7,8],[9,11,13],[15,16,17]]
Output: [15]
Explanation: 15 is the only lucky number since it is the minimum in its row and the maximum in its column

Input: matrix = [[1,10,4,2],[9,3,8,7],[15,16,17,12]]
Output: [12]
Explanation: 12 is the only lucky number since it is the minimum in its row and the maximum in its column.

Input: matrix = [[7,8],[1,2]]
Output: [7]
```
```python
class Solution:
    def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
        m = len(matrix)
        n = len(matrix[0])
        ans = []
        for i in range(m):
            min_row = min(matrix[i])
            for j in range(n):
                if matrix[i][j]==min_row:
                    max_col = max([matrix[k][j] for k in range(m)])
                    if max_col==min_row:
                        ans.append(min_row)
        return ans
```

### 1886. (easy) Determine Whether Matrix Can Be Obtained By Rotation
Given two `n x n` binary matrices `mat` and `target`, return `true` *if it is possible to make* `mat` *equal to* `target` *by **rotating** `mat` **in 90-degree increments**, or* `false` *otherwise*.
```
Input: mat = [[0,1],[1,0]], target = [[1,0],[0,1]]
Output: true
Explanation: We can rotate mat 90 degrees clockwise to make mat equal target.

Input: mat = [[0,1],[1,1]], target = [[1,0],[0,1]]
Output: false
Explanation: It is impossible to make mat equal to target by rotating mat.
```
```python
class Solution:
    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        n = len(mat)
        for _ in range(4):
            for i in range(n):
                for j in range(i,n):
                    mat[i][j],mat[j][i]=mat[j][i],mat[i][j]
            for i in range(n):
                mat[i].reverse()
            if mat==target:
                return True
        return False
```
### 11. (medium) Container With Most Water
Given `n` non-negative integers `a1, a2, ..., an` , where each represents a point at coordinate `(i, ai)`. `n` vertical lines are drawn such that the two endpoints of the line `i` is at `(i, ai)` and `(i, 0)`. Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.
```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7].
In this case, the max area of water (blue section) the container can contain is 49.
```
```python
# two pointers
class Solution:
    def maxArea(self, height: List[int]) -> int:
        # two pointers
        # O(n)
        # O(1)
        left, right = 0, len(height)-1
        max_area = 0
        while left < right:
            max_area = max(max_area, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_area
```

### 1985. (medium) Find the Kth Largest Integer in the Array
You are given an array of strings `nums` and an integer `k`. Each string in `nums` represents an integer without leading zeros.
Return the string that represents the `kth` largest integer in nums.
```python
class Solution:
    def kthLargestNumber(self, nums: List[str], k: int) -> str:
        return sorted(nums, key=int)[-k]
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
### 540. (medium) Single Element in a Sorted Array
You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once. Find this single element that appears only once.
Follow up: Your solution should run in O(log n) time and O(1) space.
```
Input: nums = [1,1,2,3,3,4,4,8,8]
Output: 2

Input: nums = [3,3,7,7,10,11,11]
Output: 10
```
```python
# binary search
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            if mid % 2 == 1:
                mid -= 1
            if nums[mid] == nums[mid + 1]:
                l = mid + 2
            else:
                r = mid
        return nums[l]
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
### 13. (easy) Roman to Integer
Given a roman numeral, convert it to an integer.
```
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```
```
Input: s = "III"
Output: 3

Input: s = "LVIII"
Output: 58

Input: s = "MCMXCIV"
Output: 1994
```

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        dic = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        ans = 0
        for i in range(len(s)):
            if i < len(s) - 1 and dic[s[i]] < dic[s[i + 1]]:
                ans -= dic[s[i]]
            else:
                ans += dic[s[i]]
        return ans
```
### 2409. (easy) Count Days Spent Together
Given two dates `date1` and `date2` in the form `MM-DD`, return the number of days between the two dates.
```python
class Solution:
    def countDaysTogether(self, arriveAlice: str, leaveAlice: str, arriveBob: str, leaveBob: str) -> int:
        a = max(arriveAlice, arriveBob)
        b = min(leaveAlice, leaveBob)
        days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        x = sum(days[:int(a[:2]) - 1]) + int(a[3:])
        y = sum(days[:int(b[:2]) - 1]) + int(b[3:])
        return max(y - x + 1, 0)
```
### 531. (medium) Lonely Pixel I
Given an `m x n` picture `consisting` of black `'B'` and white `'W'` pixels, return the number of **black** lonely pixels.
A black lonely pixel is a character `'B'` that located at a specific position where the same row and same column don't have any other black pixels.
```
Input: picture = [["W","W","B"],["W","B","W"],["B","W","W"]]
Output: 3
Explanation: All the three 'B's are black lonely pixels.
```

```python
# Brute Force
# O(mn) time, O(m+n) space
class Solution:
    def findLonelyPixel(self, picture: List[List[str]]) -> int:
        m, n = len(picture), len(picture[0])
        row, col = [0] * m, [0] * n
        for i in range(m):
            for j in range(n):
                if picture[i][j] == 'B':
                    row[i] += 1
                    col[j] += 1
        ans = 0
        for i in range(m):
            for j in range(n):
                if picture[i][j] == 'B' and row[i] == 1 and col[j] == 1:
                    ans += 1
        return ans
```

### 1003. (medium) Check If Word Is Valid After Substitutions
Given a string `s`, determine if it is valid.
A string `s` is valid if, starting with an empty string `t = ""`, you can transform `t` into `s` after performing the following operation **any number of times**:
- Insert string `"abc"` into any position in `t`. More formally, `t` becomes `tleft + "abc" + tright`, where `t == tleft + tright`. Note that `tleft` and `tright` may be empty.
- Return `true` if `s` is a valid string, otherwise, return `false`.
```python

class Solution:
    def isValid(self, s: str) -> bool:
        while 'abc' in s:
            s = s.replace('abc', '') # replace all 'abc' with ''
        return s == ''

class Sollution:
    def isValid(self, s: str) -> bool:
        stack = []
        for c in s:
            if c == 'c':
                if stack[-2:] != ['a', 'b']: # if stack[-2:] is not ['a', 'b']
                    return False
                stack.pop() # pop 'b'
                stack.pop() # pop 'a'
            else:
                stack.append(c) # append 'a' or 'b'
        return not stack
```

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
### 1208. (medium) Get Equal Substrings Within Budget
You are given two strings `s` and `t` of the same length. You want to change `s` to `t`. Changing the `i`-th character of `s` to `i`-th character of `t` costs `|s[i] - t[i]|` that is, the absolute difference between the ASCII values of the characters.
You are also given an integer `maxCost`.
Return the maximum length of a substring of `s` that can be changed to be the same as the corresponding substring of `t` with a cost less than or equal to `maxCost`.
```
Input: s = "abcd", t = "bcdf", maxCost = 3
Output: 3
Explanation: "abc" of s can change to "bcd". That costs 3, so the maximum length is 3.

Input: s = "abcd", t = "cdef", maxCost = 3
Output: 1
Explanation: Each character in s costs 2 to change to charactor in t, so the maximum length is 1.
```
```python
# sliding window
class Solution:
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        n = len(s)
        diff = [abs(ord(s[i]) - ord(t[i])) for i in range(n)] # ord() return the ASCII value of a character
        maxLen = start = end = 0
        curCost = 0
        while end < n:
            curCost += diff[end]
            while curCost > maxCost:
                curCost -= diff[start]
                start += 1
            maxLen = max(maxLen, end - start + 1)
            end += 1
        return maxLen
```

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
### 933. (easy) Number of Recent Calls
You have a `RecentCounter` class which counts the number of recent requests within a certain time frame.
Implement the `RecentCounter` class:
- `RecentCounter()` Initializes the counter with zero recent requests.
- `int ping(int t)` Adds a new request at time `t`, where `t` represents some time in milliseconds, and returns the number of requests that has happened in the past `3000` milliseconds (including the new request). Specifically, return the number of requests that have happened in the inclusive range `[t - 3000, t]`.
- It is guaranteed that every call to `ping` uses a strictly larger value of `t` than the previous call.
```python
class RecentCounter:

    def __init__(self):
        self.q = collections.deque()

    def ping(self, t: int) -> int:
        self.q.append(t)
        while self.q[0] < t-3000:
            self.q.popleft()
        return len(self.q)
```

## Tree

### 94. (easy) Binary Tree Inorder Traversal
Given the `root` of a binary tree, return *the inorder traversal of its nodes' values*.
```
Input: root = [1,null,2,3]
Output: [1,3,2]

Input: root = []
Output: []
```
```python
#Recursion
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res=[]
        def dfs(root):
            if not root: return
            dfs(root.left)
            res.append(root.val)
            dfs(root.right)
        dfs(root)
        return res

#Iteration
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res=[]
        stack=[]
        while root or stack:
            while root:
                stack.append(root)
                root=root.left
            root=stack.pop()
            res.append(root.val)
            root=root.right
        return res
```


### 965. (easy) Univalued Binary Tree
A binary tree is **univalued** if every node in the tree has the same value.
Return `true` if and only if the given tree is univalued.
```python
#Recursion
class Solution:
    def isUnivalTree(self, root: Optional[TreeNode]) -> bool:
        if not root: return True
        if root.left and root.left.val != root.val: return False #if left child exists and left child's value is not equal to root's value, return False
        if root.right and root.right.val != root.val: return False #if right child exists and right child's value is not equal to root's value, return False
        return self.isUnivalTree(root.left) and self.isUnivalTree(root.right) #recursion
```
### 107. (medium) Binary Tree Level Order Traversal II
Given the `root` of a binary tree, return *the bottom-up level order traversal of its nodes' values*. (i.e., from left to right, level by level from leaf to root).
```python
#BFS
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        res, queue = [], collections.deque([root])
        while queue: #while queue is not empty
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()  #pop the first element
                level.append(node.val)#add the first element to the level
                if node.left: #add the left child to the queue
                    queue.append(node.left)
                if node.right: #add the right child to the queue
                    queue.append(node.right)
            res.append(level) #add the level to the res
        return res[::-1]
```

### 1026. (medium) Maximum Difference Between Node and Ancestor
Given the `root` of a binary tree, find the maximum value `V` for which there exist **different** nodes `A` and `B` where `V = |A.val - B.val|` and `A` is an ancestor of `B`.
(A node `A` is an ancestor of `B` if either: any child of `A` is equal to `B`, or any child of `A` is an ancestor of `B`.)
```
Input: root = [8,3,10,1,6,null,14,null,null,4,7,13]
Output: 7
Explanation: We have various ancestor-node differences, some of which are given below :
|8 - 3| = 5
|3 - 7| = 4
|8 - 1| = 7
|10 - 13| = 3
Among all possible differences, the maximum value of 7 is obtained by |8 - 1| = 7.
```

```python
#Recursion
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        def dfs(node, cur_max, cur_min):
            if not node: return cur_max - cur_min
            cur_max = max(cur_max, node.val)
            cur_min = min(cur_min, node.val)
            left = dfs(node.left, cur_max, cur_min)
            right = dfs(node.right, cur_max, cur_min)
            return max(left, right)
        return dfs(root, root.val, root.val)
```

## Hashmap

### 1. (easy) Two Sum
Given an array of integers `nums` and an integer `target`, return *indices of the two numbers such that they add up to `target`*.
You may assume that each input would have **exactly one solution**, and you may not use the *same* element twice.
You can return the answer in any order.
```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explaination: Because nums[0] + nums[1] == 9, we return [0, 1].

Input: nums = [3,2,4], target = 6
Output: [1,2]
```
```python
#Hashmap
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic={}
        for i, num in enumerate(nums):
            if target-num in dic:
                return [dic[target-num], i]
            dic[num]=i
```
### 242. (easy) Valid Anagram
Given two strings `s` and `t`, return `true` *if `t` is an anagram of `s`, and `false` otherwise*.
```
Input: s = "anagram", t = "nagaram"
Output: true

Input: s = "rat", t = "car"
Output: false
```
```python
#Hashmap
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        dic={}
        for i in s:
            if i in dic:
                dic[i]+=1
            else:
                dic[i]=1
        for i in t:
            if i in dic:
                dic[i]-=1
            else:
                return False
        for i in dic:
            if dic[i]!=0:
                return False
        return True

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        else: 
            d = collections.defaultdict(int)
            n = len(s)
            for i in range(n):
                d[s[i]]+=1
                d[t[i]]-=1
        return all(value == 0 for value in d.values())
```

### 383. (easy) Ransom Note
Given two stings `ransomNote` and `magazine`, return `true` if `ransomNote` can be constructed from `magazine` and `false` otherwise.
Each letter in `magazine` can only be used once in `ransomNote`.
```
Input: ransomNote = "a", magazine = "b"
Output: false

Input: ransomNote = "aa", magazine = "ab"
Output: false

Input: ransomNote = "aa", magazine = "aab"
Output: true
```
```python
#Hashmap
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        dic={}
        for i in magazine:
            if i in dic:
                dic[i]+=1
            else:
                dic[i]=1
        for i in ransomNote:
            if i in dic:
                dic[i]-=1
            else:
                return False
        for i in dic:
            if dic[i]<0:
                return False
        return True
```

### 2441. (easy) Largest Positive Integer That Exists With Its Negative
Given an integer array `nums` that does not contain any zeros, find the l**argest positive** integer `k` such that `-k` also exists in the array.

Return the *positive* integer `k`. If there is no such integer, return `-1`.
```
Input: nums = [-1,2,-3,3]
Output: 3
Explanation: 3 is the only valid k we can find in the array.

Input: nums = [-1,10,6,7,-7,1]
Output: 7
Explanation: Both 1 and 7 have their corresponding negative values in the array. 7 has a larger value.

Input: nums = [-10,8,6,7,-2,-3]
Output: -1
Explanation: There is no a single valid k, we return -1.
```
```python
#Hashmap
class Solution:
    def findMaxK(self, nums: List[int]) -> int:
        ans = -1
        d = collections.defaultdict(int)
        for i in range(len(nums)):
            if d[-nums[i]] >=1:
                ans = max(ans, abs(nums[i]))
                d[-nums[i]]-=1
            else:
                d[nums[i]] += 1
        return ans
```
### 49. (medium) Group Anagrams
Given an array of strings `strs`, group **the anagrams** together. You can return the answer in **any order**.
An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
```
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Input: strs = [""]
Output: [[""]]

Input: strs = ["a"]
Output: [["a"]]
```
```python
#Hashmap
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = collections.defaultdict(list)
        for st in strs:
            key = "".join(sorted(st))
            mp[key].append(st)
        return list(mp.values())

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = collections.defaultdict(list)
        for st in strs:
            counts = [0] * 26
            for ch in st:
                counts[ord(ch) - ord("a")] += 1
            # list to tuple for hashability
            mp[tuple(counts)].append(st)        
        return list(mp.values())
```

### 1072. (medium) Flip Columns For Maximum Number of Equal Rows
Given a `matrix` consisting of 0s and 1s, we may choose any number of columns in the matrix and flip every cell in that column.  Flipping a cell changes the value of that cell from 0 to 1 or from 1 to 0.
Return the maximum number of rows that have all values equal after some number of flips.
```
Input: [[0,1],[1,1]]
Output: 1

Input: [[0,1],[1,0]]
Output: 2

Input: [[0,0,0],[0,0,1],[1,1,0]]
Output: 2
```
```python
#Hashmap
class Solution:
    def maxEqualRowsAfterFlips(self, matrix: List[List[int]]) -> int:
        cnt = Counter() # use tuple as key
        for row in matrix:
            if row[0]:  # if the first element is 1, flip the row
                for j in range(len(row)):
                    row[j] ^= 1 # XOR operation
            cnt[tuple(row)] += 1 
        return max(cnt.values())
```

## Dynamic Programming

### 413. (easy) Arithmetic Slices
An integer array is called arithmetic if it consists of **at least three elements** and if the difference between any two consecutive elements is the same.
Return the number of arithmetic subarrays of `nums`.
A **subarray** is a contiguous subsequence of the array.
```
Input: nums = [1,2,3,4]
Output: 3
Explanation: We have 3 arithmetic slices in nums: [1, 2, 3], [2, 3, 4] and [1,2,3,4] itself.

Input: nums = [1]
Output: 0
```
```python
#DP
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        for i in range(2, len(nums)):
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                dp[i] = dp[i-1] + 1
        return sum(dp)
```
### 926. (medium) Flip String to Monotone Increasing
A binary string is monotone increasing if it consists of some number of `0`'s (possibly none), followed by some number of `1`'s (also possibly none).
You are given a binary string `s`. You can flip `s[i]` changing it from `0` to `1` or from `1` to `0`.
Return *the minimum number of flips to make* `s` *monotone increasing*.
```
Input: s = "00110"
Output: 1
Explanation: We flip the last digit to get 00111.

Input: s = "010110"
Output: 2
Explanation: We flip to get 011111, or alternatively 000111.

Input: s = "00011000"
Output: 2
Explanation: We flip to get 00000000.
```
```python
#DP
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        dp0 = dp1 = 0
        for c in s:
            dp0New, dp1New = dp0, min(dp0, dp1)
            if c == '1':
                dp0New += 1
            else:
                dp1New += 1
            dp0, dp1 = dp0New, dp1New
        return min(dp0, dp1)
```

### 1024. (medium) Video Stitching
You are given a series of video clips from a sporting event that lasted `t` seconds. These video clips can be overlapping with each other and have varied lengths.
Each video clip `clips[i]` is an interval: it starts at time `clips[i][0]` and ends at time `clips[i][1]`. We can cut these clips into segments freely: for example, a clip `[0, 7]` can be cut into segments `[0, 1] + [1, 3] + [3, 7]`.
Return *the minimum number of clips needed so that we can cut the clips into segments that cover the entire sporting event* `[0, t]`. If the task is impossible, return `-1`.
```
Input: clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]], time = 10
Output: 3
Explanation:
We take the clips [0,2], [8,10], [1,9]; a total of 3 clips.
Then, we can reconstruct the sporting event as follows:
We cut [1,9] into segments [1,2] + [2,8] + [8,9].
Now we have segments [0,2] + [2,8] + [8,10] which cover the sporting event [0, 10].
```
```python
#DP
class Solution:
    def videoStitching(self, clips: List[List[int]], time: int) -> int:
        dp = [0] + [float('inf')] * time
        for i in range(1, time + 1):
            for aj, bj in clips:
                if aj < i <= bj:
                    dp[i] = min(dp[i], dp[aj] + 1)
        return -1 if dp[-1] == float('inf') else dp[-1]
```
### 1027. (medium) Longest Arithmetic Subsequence
Given an array `nums` of integers, return the *length* of the longest arithmetic subsequence in `nums`.
Recall that a *subsequence* of an array `nums` is a list `nums[i1], nums[i2], ..., nums[ik]` with `0 <= i1 < i2 < ... < ik <= nums.length - 1`, and that a sequence `seq` is *arithmetic* if `seq[i+1] - seq[i]` are all the same value (for `0 <= i < seq.length - 1`).
```

Input: nums = [3,6,9,12]
Output: 4
Explanation: The whole array is an arithmetic sequence with steps of length = 3.

Input: nums = [9,4,7,2,10]
Output: 3
Explanation: The longest arithmetic subsequence is [4,7,10].

Input: nums = [20,1,15,3,10,5,8]
Output: 4
Explanation: The longest arithmetic subsequence is [20,15,10,5].
```
thinking process: If `nums[j]` can be the last element of an arithmetic sequence with a common difference `d`, then `nums[i]` can be appended to it to form a longer arithmetic sequence. The state transition equation can be expressed as `dp[i][d] = dp[j][d] + 1`.
```python
#DP
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        dp = collections.defaultdict(int)
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                d = nums[j] - nums[i]
                dp[j, d] = max(dp[j, d], dp[i, d]+1) 
        return max(dp.values())+1
```
### 1043. (medium) Partition Array for Maximum Sum
Given an integer array `arr`, partition the array into (contiguous) subarrays of length at most `k`. After partitioning, each subarray has their values changed to become the maximum value of that subarray.
Return *the largest sum of the given array after partitioning*. Test cases are generated so that the answer fits in a 32-bit integer.
```
Input: arr = [1,15,7,9,2,5,10], k = 3
Output: 84
Explanation: arr becomes [15,15,15,9,10,10,10]

Input: arr = [1,4,1,5,7,3,6,1,9,9,3], k = 4
Output: 83

Input: arr = [1], k = 1
Output: 1
```
```python
#DP
class Solution:
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        dp = [0] * (len(arr) + 1)
        for i in range(1, len(arr) + 1):
            curMax = float('-inf')
            for j in range(1, min(i, k) + 1):
                curMax = max(curMax, arr[i - j])
                dp[i] = max(dp[i], dp[i - j] + curMax * j)
        return dp[-1]
```
### 1079. (medium) Letter Tile Possibilities
You have `n`  `tiles`, where each tile has one letter `tiles[i]` printed on it.
Return *the number of possible non-empty sequences of letters* you can make using the letters printed on those `tiles`.
```
Input: tiles = "AAB"
Output: 8

Input: tiles = "AAABBC"
Output: 188
```
```python
#DFS
class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        f = [1] + [0] * len(tiles)
        n = 0
        for cnt in Counter(tiles).values():
            n += cnt
            for j in range(n, 0, -1):
                for k in range(1, min(j, cnt) + 1):
                    f[j] += f[j - k] * comb(j, k) 
        return sum(f[1:])
```

### 1105. (medium) Filling Bookcase Shelves
link: https://leetcode.com/problems/filling-bookcase-shelves/
```python
class Solution:
    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        dp = [0] + [float('inf')] * len(books) #dp[i] is the min height of the first i books
        for i in range(1, len(dp)): 
            cur_width, max_height = 0, 0 
            for j in range(i - 1, -1, -1): #for each i, we try to put the last book on the shelf, then the second last book, and so on
                cur_width += books[j][0]
                if cur_width > shelfWidth: break 
                max_height = max(max_height, books[j][1]) 
                dp[i] = min(dp[i], dp[j] + max_height) 
        return dp[-1]
```
### 1218. (medium) Longest Arithmetic Subsequence of Given Difference
Given an integer array `arr` and an integer `difference`, return the length of the longest subsequence in `arr` which is an arithmetic sequence such that the difference between adjacent elements in the subsequence equals `difference`.
```
Input: arr = [1,2,3,4], difference = 1
Output: 4
Explanation: The longest arithmetic subsequence is [1,2,3,4].

Input: arr = [1,3,5,7], difference = 1
Output: 1
Explanation: The longest arithmetic subsequence is any single element.

Input: arr = [1,5,7,8,5,3,4,2,1], difference = -2
Output: 4
Explanation: The longest arithmetic subsequence is [7,5,3,1].
```
```python
#DP
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        dp = collections.defaultdict(int)
        for num in arr:
            dp[num] = max(dp[num], dp[num-difference]+1)
        return max(dp.values())
```

## Backtracking
## Greedy

### 1054. (medium) Distant Barcodes
In a warehouse, there is a row of barcodes, where the `ith` barcode is `barcodes[i]`.
Rearrange the barcodes so that no two adjacent barcodes are equal. You may return any answer, and it is guaranteed an answer exists.
```
Input: barcodes = [1,1,1,2,2,2]
Output: [2,1,2,1,2,1]

Input: barcodes = [1,1,1,1,2,2,3,3]
Output: [1,3,1,3,1,2,1,2]
```
```python
#Greedy
class Solution:
    def rearrangeBarcodes(self, barcodes: List[int]) -> List[int]:
        count = collections.Counter(barcodes) #count the frequency of each barcode
        res = [0] * len(barcodes)
        i = 0
        for k, v in count.most_common(): #sort the barcodes by frequency
            for _ in range(v): #put the most frequent barcode first, then the second most frequent barcode, and so on
                res[i] = k
                i += 2
                if i >= len(barcodes):
                    i = 1
        return res
```
### 1909. (medium) Remove One Element to Make the Array Strictly Increasing
Given a **0-indexed** integer array `nums`, return `true` *if it can be made strictly increasing after removing **exactly one** element, or* `false` *otherwise. If the array is already strictly increasing, return* `true`.
The array `nums` is **strictly increasing** if `nums[i - 1] < nums[i]` for each index `(1 <= i < nums.length).`
```
Input: nums = [1,2,10,5,7]
Output: true

Input: nums = [2,3,1,2]
Output: false
```
```python
#Greedy
class Solution:
    def canBeIncreasing(self, nums: List[int]) -> bool:
        n = len(nums)
        def check(idx: int) -> bool:
            for i in range(1, n - 1):
                prev, curr = i - 1, i
                if prev >= idx:
                    prev += 1
                if curr >= idx:
                    curr += 1
                if nums[curr] <= nums[prev]:
                    return False
            return True
        
        for i in range(1, n):
            if nums[i] <= nums[i-1]:
                return check(i) or check(i - 1)
        return True
```
## Simulation
### 1073. (medium) Adding Two Negabinary Numbers
Given two numbers `arr1` and `arr2` in base `-2`, return the result of adding them together.
Each number is given in `array` format:  as an array of 0s and 1s, from most significant bit to least significant bit.  For example, `arr = [1,1,0,1]` represents the number `(-2)^3 + (-2)^2 + (-2)^0 = -3`.  A number `arr` in `array`, format is also guaranteed to have no leading zeros: either `arr == [0]` or `arr[0] == 1`.
Return the result of adding `arr1` and `arr2` in the same format: as an array of 0s and 1s with no leading zeros.
```
Input: arr1 = [1,1,1,1,1], arr2 = [1,0,1]
Output: [1,0,0,0,0]

Input: arr1 = [0], arr2 = [0]
Output: [0]

Input: arr1 = [0], arr2 = [1]
Output: [1]
```
```python
#Simulation
class Solution:
    def addNegabinary(self, arr1: List[int], arr2: List[int]) -> List[int]:
        i, j, carry = len(arr1) - 1, len(arr2) - 1, 0
        res = []
        while i >= 0 or j >= 0 or carry:
            if i >= 0: carry += arr1[i]
            if j >= 0: carry += arr2[j]
            res.append(carry & 1)
            carry = -(carry >> 1)
            i -= 1
            j -= 1
        while len(res) > 1 and res[-1] == 0: res.pop()
        return res[::-1]
```

## Disjoint Set
### 1020. (medium) Number of Enclaves
You are given an `m x n` binary matrix `grid`, where `0` represents a sea cell and `1` represents a land cell.
A *move* consists of walking from one land cell to another adjacent (4-directionally) land cell or walking off the boundary of the grid.
Return *the number of land cells in* `grid` *for which we cannot walk off the boundary of the grid in any number of moves*.
```
Input: grid = [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
Output: 3
Explanation: There are three 1s that are enclosed by 0s, and one 1 that is not enclosed because its on the boundary.

Input: grid = [[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]
Output: 0
Explanation: All 1s are either on the boundary or can reach the boundary.
```
```python
#DFS
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        def dfs(i, j):
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == 1:
                grid[i][j] = 0
                dfs(i+1, j)
                dfs(i-1, j)
                dfs(i, j+1)
                dfs(i, j-1)
        for i in range(len(grid)):
            dfs(i, 0)
            dfs(i, len(grid[0])-1)
        for j in range(len(grid[0])):
            dfs(0, j)
            dfs(len(grid)-1, j)
        return sum(sum(row) for row in grid)
```

## Segment Tree