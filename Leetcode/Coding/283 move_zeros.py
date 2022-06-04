#https://leetcode.com/problems/move-zeroes/
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        l = 0
        r = len(nums) - 1
        i=0
        # while(l<=r):
        #     if nums[i]!=0:
        #         res[l]=nums[i]
        #         i+=1
        #         l+=1
        #     else:
        #         res[r]=0
        #         i+=1
        #         r-=1
        # print(res)

        array_length = len(nums)
        last_nonz = 0
        while(i<array_length):
            if(nums[i]!=0):
                nums[i],nums[last_nonz] = nums[last_nonz],nums[i]
                last_nonz+=1
            i+=1
                
            