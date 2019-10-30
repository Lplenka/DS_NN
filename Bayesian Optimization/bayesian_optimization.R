library(RSocrata)
library(dplyr)
library(lubridate)

fbi_code = "'05'"
url = sprintf("https://data.cityofchicago.org/resource/6zsd-86xi.json?$select=*&$where=fbi_code=%s", fbi_code)
burglaries = read.socrata(url)

#Converts date format from "%Y-%m-%d %H:%M:%S" to "%Y-%m-%d"
burglaries$date_clean = as.Date(as.POSIXct(burglaries$date, format = '%Y-%m-%d %H:%M:%S'))


#Creates a column "week" containing first date of the week which the column "date_clean" belongs
burglaries$week= as.Date(cut(burglaries$date_clean, 'week'))


#Find the burglary counts per district per week
burglary_counts       = burglaries %>%
  group_by(district, week) %>%
  summarise(COUNTCRIMES = length(district))
