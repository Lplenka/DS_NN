# """
# This is ArrayReader's API interface.
# You should not implement it, or speculate about its implementation
# """
#class ArrayReader:
#    def get(self, index: int) -> int:

class Solution:
    def search(self, reader: 'ArrayReader', target: int) -> int:
        
        
        max_ = 1
        min_ = 0
        
        while(min_ <= max_):
            
            if target <= reader.get(max_):
                break
            else:
                min_ = max_
                max_ = max_ * 2
            
        
        if max_ == 0 and reader.get(0) == target:
            return 0
        
        while (min_ <= max_):
            mid_ =min_ + int((max_- min_)/2)
            
            if reader.get(mid_) == target:
                return mid_
            elif reader.get(mid_) < target:
                min_ = mid_ + 1
            else:
                max_ = mid_ - 1 
                
        return -1
        