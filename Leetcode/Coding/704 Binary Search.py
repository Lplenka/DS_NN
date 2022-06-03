class Solution:
    def search(self, nums: List[int], target: int) -> int:
        max_ = len(nums)-1
        min_ = 0
        
        if max_ == 0 and nums[0] == target:
            return 0
        
        while (min_ <= max_):
            mid_ =min_ + int((max_- min_)/2)
            
            if nums[mid_] == target:
                return mid_
            elif nums[mid_] < target:
                min_ = mid_ + 1
            else:
                max_ = mid_ - 1 
                
        return -1
        
                
        