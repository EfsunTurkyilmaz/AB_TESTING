##############################################################
# Comparison of conversion of Bidding Methods with the AB Test
##############################################################

################ Dataset ####################################

# Impression: number of ad views
# Click: number of ad clicks displayed
# Purchase: number of products purchased after clicking on ads
# Earning: profit obtained from purchased products

# control group: maximum bidding
# test group: average bidding

######################## Library Imports ########################

import pandas as pd
import itertools
import statsmodels.stats.api as sms
import numpy as np
import seaborn as sns
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

################### Data Preparation and Analysis ####################

# Assign control and treatment group data to separate variables

df_c = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Control Group")
df_t = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Test Group")

# Analyze control and treatment group data

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df_c)
check_df(df_t)

# df_c.shape    ## 40 observation 4 variables
# df_t.shape    ## 40 observation 4 variables

df_c.describe().T

#               count         mean         std         min         25%         50%          75%          max
# Impression 40.00000 101711.44907 20302.15786 45475.94296 85726.69035 99790.70108 115212.81654 147539.33633
# Click      40.00000   5100.65737  1329.98550  2189.75316  4124.30413  5001.22060   5923.80360   7959.12507
# Purchase   40.00000    550.89406   134.10820   267.02894   470.09553   531.20631    637.95709    801.79502
# Earning    40.00000   1908.56830   302.91778  1253.98952  1685.84720  1975.16052   2119.80278   2497.29522

df_t.describe().T

#               count         mean         std         min          25%          50%          75%          max
# Impression 40.00000 120512.41176 18807.44871 79033.83492 112691.97077 119291.30077 132050.57893 158605.92048
# Click      40.00000   3967.54976   923.09507  1836.62986   3376.81902   3931.35980   4660.49791   6019.69508
# Purchase   40.00000    582.10610   161.15251   311.62952    444.62683    551.35573    699.86236    889.91046
# Earning    40.00000   2514.89073   282.73085  1939.61124   2280.53743   2544.66611   2761.54540   3171.48971

# Concatenate control and test group data using concat method

## Add a "group" variable to both datasets to indicate Control or Test membership

df_c["group"] = "control"
df_t["group"] = "test"

df = pd.concat([df_c, df_t], axis=0, ignore_index=True)

df.head()

#     Impression      Click  Purchase    Earning    group
# 0  82529.45927 6090.07732 665.21125 2311.27714  control
# 1  98050.45193 3382.86179 315.08489 1742.80686  control
# 2  82696.02355 4167.96575 458.08374 1797.82745  control
# 3 109914.40040 4910.88224 487.09077 1696.22918  control
# 4 108457.76263 5987.65581 441.03405 1543.72018  control

df.tail()

#      Impression      Click  Purchase    Earning group
# 75  79234.91193 6002.21358 382.04712 2277.86398  test
# 76 130702.23941 3626.32007 449.82459 2530.84133  test
# 77 116481.87337 4702.78247 472.45373 2597.91763  test
# 78  79033.83492 4495.42818 425.35910 2595.85788  test
# 79 102257.45409 4800.06832 521.31073 2967.51839  test

#####################################################
#  Define the Hypothesis for the A/B Test
#####################################################

# Define the hypothesis.

# H0: M1 = M2 (There is no difference between the mean purchases of the control and test groups)
# H1: M1 != M2 (There is a difference between the mean purchases of the control and test groups)

# Analyze the mean purchases for the control and test groups

df.groupby("group").agg({"Purchase": "mean"})

#          Purchase
# group
# control 550.89406
# test    582.10610

# The two groups appear to have different means for our success metric, "purchase",
# Perform a hypothesis test to determine if this difference is statistically significant

#####################################################
#  Conduct the Hypothesis Testing
#####################################################

# Perform assumption checks prior to hypothesis testing.
# These include the Normality Assumption and the Homogeneity of Variance.

# Evaluate whether the control and test groups satisfy the normality assumption
# with respect to the "purchase" variable, separately.

# Normality Assumption:
# H0: The data are normally distributed.
# H1: The data are not normally distributed.

# p-value < 0.05 H0 reject
# p-value > 0.05 H0 fail to reject

# for control group

test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

## Test Stat = 0.9773, p-value = 0.5891
## p = 0.5891 > 0.05  Fail to reject H0.
## Thus, the normality assumption holds for the dataset.

# for test group

test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

## Test Stat = 0.9589, p-value = 0.1541
##  p = 0.1541 > 0.05  Fail to reject H0.
## Thus, the normality assumption holds for the dataset.

# Homogeneity of Variance:

## Assess whether the homogeneity of variances assumption is satisfied
## for the Control and Test groups with respect to the "purchase" variable.

# H0: Variances are homogeneous.
# H1: Variances are not homogeneous.

# If p-value < 0.05   Reject H0.
# If p-value > 0.05   Fail to reject H0.

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Test Stat = 2.6393, p-value = 0.1083
# p = 0.1083 > 0.05  Fail to reject H0.
# Variances are homogeneous.
# The assumption of homogeneity of variances holds for both the Control and Test groups.



### Since both the normality and homogeneity of variances assumptions are satisfied,
### an Independent Samples t-test (parametric test) has been applied.


# --- Independent Samples t-test ---
# H0: M1 = M2
# There is no statistically significant difference between the mean purchase values
# of the Control and Test groups.


test_stat, pvalue =ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                             df.loc[df["group"] == "test", "Purchase"], equal_var=True)
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))


##  Test Stat = -0.9416, p-value = 0.3493
##  p-value = 0.3493 > 0.05  Fail to reject H0.
## There is no statistically significant difference between the mean purchase values of the Control and Test groups.
## The observed difference between the two group means is likely due to random chance.


#####################################################
# Interpretation of Results
#####################################################

# According to the purchase metric (the number of products purchased after clicking ads),
# there is no statistically significant difference between the Control and Test groups.

# This indicates that both methods perform similarly in terms of driving purchases.
# Therefore, either method can be confidently preferred.

