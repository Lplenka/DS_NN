library("TSA")

library("xts")
raw = read.csv("../data/weekly_dummy_data.csv")
p = periodogram(raw$Count)
dd = data.frame(freq=p$freq, spec=p$spec)
order = dd[order(-dd$spec),]
top5 = head(order, 5)

# display the 5 highest "power" frequencies
top5

# convert frequency to time periods
time = 1/top5$f
time

#raw_ts = as.ts(raw$Count)


# Convert DATE to Date class
raw$Datetime <- as.Date(as.character(raw$Datetime),format="%Y-%m-%d")
# create xts object
x <- xts(raw$Count,raw$Datetime)



autoplot(x) +
  ggtitle("Weekly data") +
  xlab("Datetime") +
  ylab("Count")


#####################################################
library(fpp2)
data("elecequip") 

elecequip %>% decompose(type="multiplicative") %>% 
  autoplot() + xlab("year") +
  ggtitle("Classical multiplicative decomposition of electrical equipment index")

elep = periodogram(elecequip)
eledd = data.frame(freq=elep$freq, spec=elep$spec)
ele_order = eledd[order(-eledd$spec),]
ele_top5 = head(ele_order, 5)

# display the 5 highest "power" frequencies
ele_top5

# convert frequency to time periods
ele_time = 1/ele_top5$f
ele_time

########################################################
data("austres") 

austres %>% decompose(type="multiplicative") %>% 
  autoplot() + xlab("year") +
  ggtitle("Classical multiplicative decomposition of Australian residents")

ausp = periodogram(austres)
ausdd = data.frame(freq=ausp$freq, spec=ausp$spec)
aus_order = ausdd[order(-ausdd$spec),]
aus_top5 = head(aus_order, 5)

# display the 5 highest "power" frequencies
aus_top5

# convert frequency to time periods
aus_time = 1/aus_top5$f
aus_time