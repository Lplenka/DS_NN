# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        max_ = n
        min_ = 1
        
        if max_ == 1 and isBadVersion(1) == True:
            return 1
        
        while (min_ <= max_):
            mid_ =min_ + int((max_- min_)/2)
            
            if isBadVersion(mid_) == True and isBadVersion(mid_- 1)== False:
                return mid_
            elif isBadVersion(mid_) == False:
                min_ = mid_ + 1
            else:
                max_ = mid_ - 1 
                
        return -1