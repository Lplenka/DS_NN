#install.packages("fpp")
#Step 1: Import the Data
library("TSA")
library(fpp)
data(ausbeer)
timeserie_beer = tail(head(ausbeer, 17*4+2),17*4-4)
plot(as.ts(timeserie_beer))
#Step 2: Detect the Trend
library(forecast)
trend_beer = ma(timeserie_beer, order = 4, centre = T)
plot(as.ts(timeserie_beer))
lines(trend_beer)
plot(as.ts(trend_beer))

#Detrend the Time Series


detrend_beer = timeserie_beer - trend_beer
plot(as.ts(detrend_beer))


#Average the Seasonality

m_beer = t(matrix(data = detrend_beer, nrow = 4))
seasonal_beer = colMeans(m_beer, na.rm = T)
plot(as.ts(rep(seasonal_beer,16)))


p = periodogram(na.remove(detrend_beer))
dd = data.frame(freq=p$freq, spec=p$spec)
order = dd[order(-dd$spec),]
top2 = head(order, 2)

# display the 5 highest "power" frequencies
top2

# convert frequency to time periods
time = 1/top2$f
time

