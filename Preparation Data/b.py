
#How many Days
from datetime import datetime

# Define the start and end dates
start_date = datetime(1976, 3, 1)
end_date = datetime(2019, 12, 31)

# Calculate the difference in days
difference = (end_date - start_date).days
print("Number of Days = ",difference+1)


#How many months


# Calculate the difference in months
from dateutil.relativedelta import relativedelta

# Compute the difference in months
difference_months = relativedelta(end_date, start_date)
total_months = difference_months.years * 12 + difference_months.months
print("number of months = ", total_months+1)
