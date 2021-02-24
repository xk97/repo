from typing import List

def fun(N):
    a, b = 1, 1
    for i in range(2, N ):
        a, b = b, a + b
    print(a + b)

[1,1, 2, 3, 5, 8]
fun(2)
fun(3)
fun(4)
fun(5)


[2, 3, 5, 10]  # 16

def coin_change( bill = [1, 4, 5], n = 17 ):
    """
    return min number of bills to reach n
    """
    arr = [0] * (n + 1)
    for i in range(1, n + 1):
        arr[i] = 1 + min([arr[i - x] for x in bill if i >= x])
    print(list(zip(range(0, n + 1), arr)))
    print(arr[-1])
coin_change()

import numpy as np
arr = np.random.permutation(range(10))
print(arr)

def longest_increase_sequence(arr):
    ml = [1] * len(arr)  # len of LIS ending with arr[i]
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[i] > arr[j]:
                ml[i] = max(ml[i], ml[j] + 1)
    print('LIS', max(ml))

longest_increase_sequence(arr)

def max_sum_subarray2(arr):
    m = s = arr[0]
    for i in range(1, len(arr)):
        if s > 0: # current sum positive and helps i
            s = s + arr[i]
        else:
            s = arr[i]
        m = max(m, s)  # global max so far
    return m

def max_sum_subarray(arr):
    m = s = arr[0]
    for i in range(1, len(arr)):
        s = max(s, s + arr[i])  # sum of current or add arr[i]
        m = max(s, m) # max of global vs current
    print('max subarray', m)
max_sum_subarray(arr)


# 3 sum = 0
"""
# 思路
标签：数组遍历
首先对数组进行排序，排序后固定一个数 nums[i]nums[i]，再使用左右指针指向 nums[i]nums[i]后面的两端，数字分别为 nums[L]nums[L] 和 nums[R]nums[R]，计算三个数的和 sumsum 判断是否满足为 00，满足则添加进结果集

作者：guanpengchn
链接：https://leetcode-cn.com/problems/3sum/solution/hua-jie-suan-fa-15-san-shu-zhi-he-by-guanpengchn/
"""
def three_sum(nums, k):
    """
    docstring
    """
    nums = sorted(nums)
    for i in range(len(nums) - 1):
        l, r = i + 1, len(nums) - 1
        while l < r:
            if nums[i] + nums[r] + nums[l] < k:
                l += 1
            elif nums[i] + nums[r] + nums[l] > k:
                r -= 1
            else:
                print('three sum', nums[i] , nums[r] , nums[l] )
                return

three_sum(list(range(10)), 14)
"""
给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。

说明：
初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。
你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
链接：https://leetcode-cn.com/problems/merge-sorted-array
"""

def merge_two_arr(nums1, nums2):
    # 
    m, n = len(nums1) - len(nums2), len(nums2)
    i, j, k = m - 1, n - 1, m + n - 1
    while i >= 0 and j >= 0:
        if nums2[j] >= nums1[i]:
            nums1[k] = nums2[j]
            j, k = j - 1, k - 1
        else: # nums1 bigger
            nums1[k] = nums1[i]
            i, k = i - 1, k - 1
        if i == -1 and j > 0:
            while j >= 0:
                nums1[j] = nums2[j]
                j -= 1
    print('merge two sorted arr', m, n, nums1)

merge_two_arr(list(range(4, 9)) + [0] * 4, list(range(4)))
merge_two_arr(list(range(0, 5)) + [0] * 4, list(range(4)))

"""
Rearrange an array in maximum minimum form | Set 2 (O(1) extra space)
Given a sorted array of positive integers, rearrange the array alternately i.e first element should be the maximum value, second minimum value, third-second max, fourth-second min and so on.

Examples:

Input  : arr[] = {1, 2, 3, 4, 5, 6, 7} 
Output : arr[] = {7, 1, 6, 2, 5, 3, 4}

Input  : arr[] = {1, 2, 3, 4, 5, 6} 
Output : arr[] = {6, 1, 5, 2, 4, 3}
"""
# even index : remaining maximum element.
# odd index  : remaining minimum element.
 
# max_index : Index of remaining maximum element
#             (Moves from right to left)
# min_index : Index of remaining minimum element
#             (Moves from left to right)


"""
# Standard partition process of QuickSort().  
# It considers the last element as pivot and 
# moves all smaller element to left of it 
# and greater elements to right 
"""
def partition(arr, k):
    x = arr[k]
    print(arr)
partition(arr=[6, 1, 5, 2, 4, 3], k=4)


# Subset Sum Problem
"""
Input: set[] = {3, 34, 4, 12, 5, 2}, sum = 9
Output: True  
There is a subset (4, 5) with sum 9.

Input: set[] = {3, 34, 4, 12, 5, 2}, sum = 30
Output: False
There is no subset that add up to 30.
"""
def is_subset_sum(arr, n, s):
    if s == 0:
        return True
    if n == 0:
        return False
    if arr[n - 1] > s:
        return is_subset_sum(arr, n - 1, s) 
    return is_subset_sum(arr, n - 1, s) | is_subset_sum(arr, n - 1, s - arr[n - 1])

print('is sub sum', is_subset_sum(arr=range(9), n=9, s=18))


class Solution:
    def maxProduct_n(self, nums ) -> int:
        mx_curr = mn_curr = mx = nums[0]
        for i in range(1, len(nums)):
            mx_prev = mx_curr
            mx_curr = max(nums[i], max(mx_prev * nums[i], mn_curr * nums[i]))
            mn_curr = min(nums[i], min(mx_prev * nums[i], mn_curr * nums[i]))
            mx = max(mx_curr, mx)
        print('max product ', mx, nums)
        return mx
        
    def maxProduct_n2(self, nums ) -> int:
        mx = s = nums[0]
        for i in range(0, len(nums)):
            s = 1
            for j in range(i, len(nums)):
                s *= nums[j]
                mx = max(mx, s)
        return mx

Solution().maxProduct_n([2,3,-2,4])
Solution().maxProduct_n2([2,3,-2,4])