import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the data 
basepath = "C:\\Gaurav Work\\ML\\machine-learning\\all_datasets_collection\\"
filename = "haberman.csv"
data = pd.read_csv(basepath + filename)

#how many datapoint and columns are there
print(data.shape)
# Observation - Data has 306 data points and 4 feature containng 3 independent variables and 1 dependent variables

# what are the columns as -
print(data.columns)
# Observation - names of the coumns are 
    # age_of_patient
    # year_of_operation
    # axillary_nodes_detected
    # survival_status



#how many data points we have for each class
# class 1 - survive 5 pr more years
# class 2 - survive less 5 years or died 
print(data["survival_status"].value_counts())
# Observtion - 225 data points are poistive means 73% of people survived more than 5 years

print(data.describe())
# Observations -
# Min and Max age of people undergone surgery is 30 and 83 respectively
# The range of data collections is from year 58 to 69
# maximum 52 axillary nodes were found 

data.plot(kind="scatter", x="age_of_patient", y = "survival_status")
plt.show()

data.plot(kind="scatter", x="age_of_patient", y = "year_of_operation")
plt.show()

#lets make it colorful
#sns.set_style('darkgrid')
sns.FacetGrid(data, hue="survival_status", size=10) \
.map(plt.scatter, "age_of_patient", "year_of_operation") \
.add_legend()
plt.show()


sns.pairplot(data, hue="survival_status", vars=["age_of_patient", "year_of_operation", "axillary_nodes_detected"])\
    .add_legend()