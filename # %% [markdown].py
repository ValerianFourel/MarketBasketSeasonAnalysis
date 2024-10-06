# %% [markdown]
# # Intro to Market Basket Analysis
# 
# This project uses a Kaggle-published open-source dataset for Market Basket Analysis, a technique that identifies patterns in frequently purchased items to inform business decisions. We'll analyze the dataset using the apriori algorithm to study how seasonality affects customer grocery habits, starting with data cleaning.
# 
# The original data can be found at: 
# https://www.kaggle.com/code/heeraldedhia/market-basket-analysis-using-apriori-algorithm/input
# 

# %% [markdown]
# ## Part 1: Exploring the Data
# 
# The first part of the project is to explore the data, in order to obtain, 4 preprocess sets divided by seasonality on which it will be possible to use the Transaction Encoder.
# We must first create the Time Frame, we will study, we will first get it by season. We will infer whether, there are difference in results given the 4 different periods that we consider.
# We can then compare the application of the apriori algorithm on all seasons.

# %%
!pip install mlxtend
import numpy as np # we must import our packages for data pre-processing
import pandas as pd # we bring pandas which will be used extensively
from mlxtend.preprocessing import TransactionEncoder # we use the packages for the apriori algorithm
from mlxtend.frequent_patterns import apriori, association_rules # we bring in the association rule
import matplotlib.pyplot as plt
import seaborn as sns

seasons = 4 #Global variable

# %% [markdown]
# Questions to answer:
# - What does TransactionEncoder do? 
# - How does TransactionEconder work?
# - What is the apriori Algorithm?
# - What are association Rules?

# %%
#data_basket = pd.read_csv("basket.csv") # bring the data
data = pd.read_csv("GroceriesData.csv") # we add our data into a frame work
data = pd.DataFrame(data) # bring the data in a dataFrame

# %%
def seasonEncoding():
    seasons = [month%12 // 3 + 1 for month in range(1, 13)] # we get the number seasonality
    # by number first
    seasonResult = []
    for season in seasons:
        if season ==1:
            seasonResult.append("Winter") # we then append the season to a list matching the
            # index
        elif season ==2:
            seasonResult.append("Spring")
        elif season ==3:
            seasonResult.append("Summer")
        else:
            seasonResult.append("Fall")
    month_to_season = dict(zip(range(1,13), seasonResult)) # we get the dictionary from the
    # month number to the season
    return month_to_season
    

# %%
month_to_season = seasonEncoding()
month_to_season # we see the dictionary encoding the Season

# %%
data.head() # this is the "raw" data

# %%
print(data['month'][0]) # the 7th month is July
month_to_season.get(data['month'][0]) #July is a summer month

# %%
data['season'] = data['month'].apply(lambda month: month_to_season.get(month))
# WE add here the season to our Data frame, so we can then separate the transaction

# %%
data.head() # the season is present

# %%
g = data.groupby(by='season') # we group by season 

# %%
data_basket = data.drop(columns=['year','month','day'],axis=1)

# %%
data_basket.head() # this is what we want for 4 different dataframe one for each season 

# %%
data_basket.fillna('!',inplace=True) # we must preprocess the data like that for the basket to be able to use mlxtend
trans = data_basket.values.tolist() 
data_basket.head() # this is what our data must look like after being transformed

# %%
dataSummer = data[data['season'] == 'Summer'] # for each season we encode it
dataWinter = data[data['season'] == 'Winter'] # for each season wer divide the data
dataFall = data[data['season'] == 'Fall']
dataSpring = data[data['season'] == 'Spring']

# %% [markdown]
# We want to quickly visually inspect that our dataframe for the Summer is correct.

# %%
print(dataSummer.shape)
dataSummer# we verify what it looks like before transforming it into a basket of transaction

# %%

grp = dataSummer.sort_values(by=['Member_number', 'Date']) # we sort the 
# dataframe of summer by member ID and Date in order to later join the itemDescription

dataSummer.groupby(['Member_number', 'Date'],group_keys=True).apply(lambda x: x)


# %% [markdown]
# We must note that transaction concerns One Customer ata specific date, and not specific shopping trip.

# %%
dataSummerTemp = dataSummer.copy() # we copy the Summer months for exploring the Data
dataSummerTemp

# %%

dataSummerTemp = dataSummerTemp.sort_values(by=['Member_number', 'Date']) 
# we sort by Member_number first and by date second

dataSummerTemp['itemDescription'] = dataSummerTemp.groupby(['Member_number', 'Date'])['itemDescription'].transform(lambda x: ','.join(x))
# we must do the costly operation of joining, all of the transaction together

dataSummerTemp = dataSummerTemp[['Member_number','Date','itemDescription']].drop_duplicates()
# wethen drop all of the duplicates creating, to keep only one row per transaction

dataSummerTemp # the temporary data yields, a table with a Member ID, a date, and what was bought 
# by the customer on that day


# %% [markdown]
# # Part 2: Creating our Basket for Analysis
# 
# We have finished the first step of preprocessing our data, up to creating a  pandas framework, where for each transaction with a row with a full transaction record of the items.

# %%

def getSeasonBasket(dataSeason): # we can move forward by applying the trnasformation frtom above to each 
    # subset of the Groceries data divided by Seasons 
    
    assert isinstance(dataSeason, pd.DataFrame)
    dataSeasonTemp = dataSeason.copy()
    dataSeasonTemp = dataSeasonTemp.sort_values(by=['Member_number', 'Date'])
    maxValue = dataSeasonTemp[['Member_number','Date','itemDescription']].groupby(['Member_number', 'Date']).count().max()
    dataSeasonTemp['itemDescription'] = dataSeasonTemp.groupby(['Member_number', 'Date'])['itemDescription'].transform(lambda x: ','.join(x))

    dataSeasonTemp = dataSeasonTemp[['Member_number','Date','itemDescription']].drop_duplicates()
    maxVal = maxValue['itemDescription'] # we get the size of the biggest transaction
    
    itemList = []
    for i in range(maxVal):
        stringItem = 'Item{item}'.format(item = i)
        itemList.append(stringItem)
    
    dfSeason = pd.DataFrame(columns=itemList) # we create dataFrame for the summer 
    dfSeason[itemList] = dataSeasonTemp['itemDescription'].apply(lambda x: pd.Series(str(x).split(",")))
    dfSeason.fillna('!',inplace=True)

    return dfSeason
    


# %%
maxValue = dataSummerTemp[['Member_number','Date','itemDescription']].groupby(['Member_number', 'Date']).count().max() 
# the max Value returns the transaction with the largest number of items for the Summer season 
# We start by Summer 

maxVal = maxValue['itemDescription'] # we get the biggest transaction
def itemListCount(maxVal): 
    itemList = []
    for i in range(maxVal):
        stringItem = 'Item{item}'.format(item = i)
        itemList.append(stringItem) # we count the number of items to make a fitted table
    return itemList


itemList = itemListCount(maxVal) # We make a list of the item number

dfSummer = pd.DataFrame(columns=itemList) # we create dataFrame for the summer 
dfSummer[itemList] = dataSummerTemp['itemDescription'].apply(lambda x: pd.Series(str(x).split(",")))
dfSummer.head()

# %%
dataSummer

# %%
basketSpring = getSeasonBasket(dataSpring) # we get the Item for each data Set
basketSummer = getSeasonBasket(dataSummer)
basketFall = getSeasonBasket(dataFall)
basketWinter = getSeasonBasket(dataWinter)
basketWinter.head()


# %%
fullBasket = pd.concat([basketSpring,basketSummer,basketFall,basketWinter])
fullBasketLength = len(fullBasket.index) # we need this to compute the probabilities later on
fullBasket.fillna('!',inplace=True)
fullBasket # we will need this value for the sampling of the chi-Square

# %% [markdown]
# # Part 3: The apriori algorithm
# 
# We can move on next apply the Apriori Algorithm from mlxtend package for each of the seasons. From here you can transform the data, in order to see the tentative results, that can be expected by using the mlxtend library.
# ### What we need to do next:
# 
# 
# We must take our 4 subset of the basket, and we must cross validate that the basket are the same, by looping for all. 

# %%
transSummer = fullBasket.values.tolist() # we must transform our pandas dataframe to a list of list 
# containg the items of each transaction

# %%
for i in range(len(transSummer)):
    transSummer[i] = [x for x in transSummer[i] if not x=='!']
transSummer

# %%
te = TransactionEncoder() # we use the Transaction Encoder of mlxtend for it
te_ary = te.fit(transSummer).transform(transSummer)
transactions = pd.DataFrame(te_ary, columns=te.columns_) # encode the transaction

# %%
freq_items = apriori(transactions, min_support=0.00025, use_colnames=True, verbose=1)
freq_items.head(7)

# %%
freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))

# %%
number_of_items = freq_items.shape[0]
freq_items.sort_values('length',ascending=False) # we get the number of times the number of itemsets' frequency

# %%
number_of_items

# %%
rules = association_rules(freq_items, metric="lift", min_threshold=0.001)
rules.head() 

# %%
rules.sort_values('lift', ascending=False) # this is what our rules must look 
# like when applied using the mlxtend package

# %% [markdown]
# # Part 4: Building our 4 Sets of Association Rules
# 
# We can now move on to building our 4 sets of association rules, one for each season. We do by building a custom function with which we can make our association rule analysis. 
# 
# It is important to note that we first want to fixate the minimum support an association rule needs to be considered in our set. The smaller our minimum support threshold, the more rule we will get and the less siginificant will be our findings. 
# 
# 

# %%
Spring = 0 # we encode each of our Season by it's happeing per months
Summer = 1
Fall   = 2
Winter = 3

# %%
min_support = 0.001 # HyperParameters
min_threshold = 0.001

# %%
def analysisMaker(dataFrameBasket,min_support=min_support,min_threshold=min_threshold):
    dataFrameBasket.fillna('!',inplace=True)
    valuesBasket = dataFrameBasket.values.tolist()
    for i in range(len(valuesBasket)):
        valuesBasket[i] = [x for x in valuesBasket[i] if not x=='!']

    te = TransactionEncoder()
    te_ary = te.fit(valuesBasket).transform(valuesBasket)
    transactions = pd.DataFrame(te_ary, columns=te.columns_)

    freq_items = apriori(transactions, min_support=min_support, use_colnames=True, verbose=1)
    freq_items.head(7)
    print('apriori:', freq_items.shape)

    freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))
    freq_items.sort_values('length',ascending=False)

    rules = association_rules(freq_items, metric="lift", min_threshold=min_threshold)
    rules.head()
    print('rules:', rules.shape)

    rules.sort_values('consequent support', ascending=False)
    return rules

# %%
baskets = np.array([basketSpring,basketSummer,basketFall,basketWinter],dtype=object)
# first consider our baskets, that is an array of basket, 1 per season

# %%

def rulesCV(baskets,min_support=min_support,min_threshold=min_threshold): # we make CV fold of the association rules
    
    rulesMainFold = [] # this will be the list of our MainFolds
    rulesTestFold = [] # this will the list of our Test Folds
    for basket in range(len(baskets)):
        concatenatedBasket = pd.concat(np.array(np.delete(baskets, basket),dtype=object))
        # concatenate our "MainFold"
    
        print(tuple(map(sum, zip(concatenatedBasket.shape, baskets[basket].shape))))
        # this print statement helps us visualize that all transactions are taken into the account in the list makers
        mainFold = analysisMaker(concatenatedBasket,min_support=min_support,min_threshold=min_threshold)
        testFold = analysisMaker(baskets[basket],min_support=min_support,min_threshold=min_threshold) 
        rulesMainFold.append(mainFold) # we append the Main Fold for each season to the Main Fold list
        rulesTestFold.append(testFold)
        
  
    return rulesMainFold, rulesTestFold


# %%
# val = analysisMaker(basketFall)
# val.sort_values('support', ascending=False)

rulesMainFold, rulesTestFold = rulesCV(baskets)

# %%
rulesMainFold[0].sort_values(by='consequent support', ascending=False) # this is our main fold

# %%
rulesTestFold[0] # this is our test fold, we see that some association rules are present in both 

# %% [markdown]
# # Part 5: Chi-Square Testing
# 
# 
# We must now do chi-square testing to see whether we have siginificant differences between the different rules. obtained, and see if there's bundles that ùre affected by seasonality

# %% [markdown]
# how do I test the p-value on the apriori algorithm?
# 
# 
# The p-value is a measure of statistical significance that is used to determine whether the results of a hypothesis test are reliable. In the context of the Apriori algorithm, the p-value is used to determine the support for each itemset in the dataset.
# 
# To test the p-value for an itemset using the Apriori algorithm, you can follow these steps:
# 
# Set a minimum support threshold (min_support) and a minimum confidence threshold (min_confidence). The min_support threshold is the minimum number of occurrences of an itemset in the dataset for it to be considered "frequent", and the min_confidence threshold is the minimum probability that an itemset will be included in the final set of frequent itemsets.
# 
# Use the Apriori algorithm to find all frequent itemsets in the dataset that meet the min_support threshold.
# 
# For each frequent itemset, calculate the support and confidence of the itemset. The support of an itemset is the percentage of transactions in the dataset that contain the itemset, and the confidence of an itemset is the probability that an itemset will be included in the final set of frequent itemsets.
# 

# %%
n = number_of_items # we will need the number of transactions for computing the Chi-Square Value
n

# %%
rulesMainFold[0][['antecedents','consequents']] # this is the list of the rules of the main fold

# %%
def getDiffOverlap(rulesMainFold,rulesTestFold):
    dfSeason_DiffInTest = [] # dataframe where there's rule in Test not in 
     # main 
    dfSeason_DiffInMain = [] # dataframe where There's rule in Main not
    # in Test
    dfSeason_overlap = [] # dataframe where there's the rule present in 
    # both 


    for value in range(seasons):
        dfSeason_diff = pd.concat([rulesMainFold[value][['antecedents','consequents']], # we concatenate the rules
                                        rulesTestFold[value][['antecedents','consequents']]]).drop_duplicates(keep=False)

        dfSeason_overlap.append(pd.merge(rulesMainFold[value][['antecedents','consequents']],
                                     rulesTestFold[value][['antecedents','consequents']], 
                                     how='inner', on=['antecedents','consequents'])) # we append the overlap
                                    # using 'fake' SQL, in pandas

        dfSeason_DiffInMain.append(pd.merge(rulesMainFold[value][['antecedents','consequents']],
                                     dfSeason_diff,
                                     how='inner', on=['antecedents','consequents']))
                # we use this same fake SQl to fetch the rules in Main not in Test

        dfSeason_DiffInTest.append(pd.merge(dfSeason_diff,
                                     rulesTestFold[value][['antecedents','consequents']],
                                     how='inner', on=['antecedents','consequents']))
            # finally we get the rules that are in the tested season, but are not in the Main Fold

    return dfSeason_DiffInTest, dfSeason_DiffInMain, dfSeason_overlap




# %%
dfSeason_DiffInTest, dfSeason_DiffInMain, dfSeason_overlap = getDiffOverlap(rulesMainFold,rulesTestFold)
# we use the function Season

# %%
dfSeason_DiffInTest[0].shape[0]+dfSeason_overlap[0].shape[0]

# %%
rulesTestFold[0].shape[0] # we check if the numbers match

# %%
dfSeason_DiffInMain[0].shape[0]+dfSeason_overlap[0].shape[0]

# %%
rulesMainFold[0].shape[0] # the numbers match here too

# %%
for val in range(seasons):
    print(rulesTestFold[val].shape[0]-rulesMainFold[val].shape[0]) # we print the difference in cardinalities
    # between the test and the main fold, we see that the test may have more or jsut as much association rules

# %%
rulesTestFold[0] # this is the results for the test rule

# %%
df = rulesMainFold[0]
df.loc[df['leverage'] > 0.001]

# %%
def ObservedContingencyTableRule(n,support, lift, confidence): # we build a list 
    # containg the probabilitie of the Observed Contigency Table
    AandB = n*support
    AandNotB = n*support/confidence*(1-confidence)
    NotAandB = n*(confidence/lift-support)
    NotAandNotB = n*(1-support/confidence*(1-confidence)-confidence/lift)
        # those probabilities are computed as Specfied in the Boston College Paper

    return [AandB,AandNotB,NotAandB,NotAandNotB]
    

# %%
def ChiSquareComputation(listObservedProbs,listExpectedProbs):
    assert len(listObservedProbs) == len(listExpectedProbs)
    # this is the main Chi_Square computation, the most useful, we will need it
    # to compare the set of rules
    # this is the computation for the Comparison
    ChiSquareValue = 0
    for value in range(len(listObservedProbs)):
        ChiSquareValue += ((np.power((listObservedProbs[value]-listExpectedProbs[value]), 2))/(listExpectedProbs[value]))
        # we use the Chi-Square formula on each 4 of the probabilities, and
        #compare the observation, the test fold season, to the main, fold, the observed, the rest of the year
    return ChiSquareValue

# %%

antecedents = dfSeason_overlap[0]['antecedents'].iloc[1] # we get the antecents values
consequents = dfSeason_overlap[0]['consequents'].iloc[1] # we get the consequent vaues
# those come from the overlap table, the intermediate table, to see
# if the values are presernt in both tables


print(antecedents) # the antcents and consequents act the unique Key of a SQL, table
print(consequents)

MainFold = rulesMainFold[0]
TestFold = rulesTestFold[0] # we get the first season

MainRow = MainFold.loc[(MainFold['antecedents'] == antecedents) & (MainFold['consequents'] == consequents)]
# we get the row from the Main Fold
MainRow


# %%
TestRow = TestFold.loc[(TestFold['antecedents'] == antecedents) & (TestFold['consequents'] == consequents)]
# we get the row from the test Fold
TestRow

# %%
support = TestRow['support'].item() # we get our 3 KPIs
lift = TestRow['lift'].item()
confidence = TestRow['confidence'].item()

lengthBasketSpring = len(baskets[0].index) #we get the number of samples, ie, the 
#number of transactions in the spring

listObservedFrequency = ObservedContingencyTableRule(lengthBasketSpring,support,lift,confidence)
# wecompute the list of the 4 probabilities as observed, the test
listObservedFrequency

# %% [markdown]
# ### Chi Square Note
# 
# We can see that we obtain unnormal values for the Chi-Sqaure, using contigency tables, this is why we need a a way to compute homogeneity, with a high number of samples. That is the Breslow-Day Test.

# %%
!pip install statsmodels
import statsmodels.stats.contingency_tables as contingency_tables # we import this package 
# for the Breslow Day Test

# %%
# we now loop through the overlaps for each test season

def AssociationRulesClassification(dfSeason_overlap,rulesMainFold,rulesTestFold):
    for season in range(seasons):
        df = dfSeason_overlap[season]
        df.reset_index()
        # we add those value
        # df['Chi-Square-Test'] = 0 # those values are just to see if there's an issue with the data
        # df['Chi-Square-Main'] = 0
                # Confidence is the KPIs to underatand how the association rule is doing 
 
        df['Confidence Ratio'] = 0   # !!!!!!!! KEY VALUE FOR COMPARISON 
        df['Support Ratio'] = 0 
        df['Lift Ratio'] = 0
        observedTestLengthN = len(baskets[season].index)
        observedMainLengthN = fullBasketLength - observedTestLengthN
        print(observedTestLengthN,observedMainLengthN,fullBasketLength) # we get the basket as
        # we need to consider all samples for our analysis


        for rules in range(len(dfSeason_overlap[season])): # we loop through the rules
            antecedents = dfSeason_overlap[season]['antecedents'].iloc[rules] # we get the antecedent
            consequents = dfSeason_overlap[season]['consequents'].iloc[rules] # we get the consequent
            MainFold = rulesMainFold[season] # we get the main fold of the season
            TestFold = rulesTestFold[season] # we get the test rules

            MainRow = MainFold.loc[(MainFold['antecedents'] == antecedents) & (MainFold['consequents'] == consequents)]
            # we get the row in the main fold associated with this rule
            TestRow = TestFold.loc[(TestFold['antecedents'] == antecedents) & (TestFold['consequents'] == consequents)]
            # we get the row in the test fold associated with this rule

            supportTest = TestRow['support'].item() # we get the test support 
            liftTest = TestRow['lift'].item() # test lift 
            confidenceTest = TestRow['confidence'].item() # test confidence

            listObservedFreq = ObservedContingencyTableRule(observedTestLengthN,supportTest,liftTest,confidenceTest) 

            
            supportMain = MainRow['support'].item()
            liftMain = MainRow['lift'].item()
            confidenceMain = MainRow['confidence'].item()


            # df.loc[rules, ['Chi-Square-Main']]= value2
            # df.loc[rules, ['Difference Support']]= supportTest - supportMain 
            # we can compare if the rule did better 

            df.loc[rules, ['Confidence Ratio']]= confidenceTest/confidenceMain #the confidence ratio 
            # this KPIs indicates if the rule, did better or worse in test season
            df.loc[rules, ['Support Ratio']]= supportTest/supportMain #
            df.loc[rules, ['Lift Ratio']]= liftTest/liftMain #


            listObservedFreqMain= ObservedContingencyTableRule(observedMainLengthN,supportMain,liftMain,confidenceMain)
            # we get the observed frequencies of the main fold, 
            # between the Observed and Expected values
            listObservedFreq.extend(listObservedFreqMain) # we get all of the frequencies

            dataContigency = np.array(listObservedFreq)
            dataContigency = dataContigency.reshape(2,2,2) # we reshape the frequencies in a numpy array

            Contigency = contingency_tables.StratifiedTable(dataContigency)
            testValues= Contigency.test_equal_odds(adjust=True) # we get the chi square and p value of the
            # Breslow Day test 
            finalVal = np.float32(testValues.statistic) # we get the value from the Bunch object
            df.loc[rules, ['Chi-Square-Unbiased']] = finalVal # this is our un-Baised-Chi-Square

        

# %%
AssociationRulesClassification(dfSeason_overlap,rulesMainFold,rulesTestFold)

# %%
dfSeason_overlap[0].loc[dfSeason_overlap[0][['Chi-Square-Unbiased']].idxmax()] # we get the highest Chi Suqare in 
# Spring

# %%
# dfSeason_overlap[1].loc[dfSeason_overlap[1][['SE']].idxmax()] # highest Chi square in summer

# %%
dfSeason_overlap[2].loc[dfSeason_overlap[2][['Chi-Square-Unbiased']].idxmax()]

# %%
dfSeason_overlap[3].loc[dfSeason_overlap[3][['Chi-Square-Unbiased']].idxmax()]

# %%
dfSeason_overlap[2].loc[dfSeason_overlap[2][['Confidence Ratio']].idxmax()]

# %%
TestFold = rulesTestFold[3] # we get the test rules

df = rulesTestFold[3].loc[rulesTestFold[3][['lift']].idxmax()]
prob_milkWinter = df['confidence']/df['lift']
df2 = rulesMainFold[3].count()
df

# %%
TestFold = rulesMainFold[3] # we get the test rules

df = rulesMainFold[3].loc[rulesMainFold[3][['lift']].idxmax()]
prob_milkWinter = df['confidence']/df['lift']
df2 = rulesMainFold[3].count()
df

# %% [markdown]
# ## Chi-Square Computation
# 
# We must write note that the Chi-Square in this figure has a single degree of freedom, as the rule A->B is considered a random binary variable, (on whether it exist or not). Therefore, we 

# %% [markdown]
# # Part 6: Analysis of the rules
# 
# We can now separate the rules into 4 categories:
# - 
# 
# It is important to note that we use The Critical value of the Chi-Square given we have 1 degree of freedom, when using it on association rule set, as explain by Sergio Alvarez. 

# %%
ChiSquareCriticalValue90 = 2.706 # 0.9 STATISTICAL SIGNIFICANCE
ChiSquareCriticalValue95 = 3.841 # 0.95 STATISTICAL SIGNIFICANCE

# %%
df = dfSeason_overlap[0]
df.loc[(df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue95)].count()

# %%
df = dfSeason_overlap[1]
df.loc[(df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue95)].count()

# %%
df = dfSeason_overlap[2]
df.loc[(df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue95)].count()

# %%
df = dfSeason_overlap[3]
df.loc[(df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue95)].count()

# %%
finalTestRules = [] # we now see the overlap and difference in the statistically siginificant 
#test result
finalMainRules = []
for value in range(seasons):
    df = dfSeason_overlap[value]
    df2 = df.loc[(df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue95)] # we make again a Test Fold
    finalTestRules.append(df2)
    mainRulesTemp = []
    for season in range(seasons):
        if season != value:
            df = dfSeason_overlap[season]
            df2 = df.loc[(df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue95)] # main fold
            mainRulesTemp.append(df2)
    
    df = pd.concat(mainRulesTemp)
    finalMainRules.append(df)
    

# %%
CheckTheCount = []
for value in range(seasons):
    df = dfSeason_overlap[value]
    df2 = df.loc[(df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue95)] # we make again a Test Fold
    CheckTheCount.append(df2)
    
df = pd.concat(CheckTheCount)

df.count() # 124 signficant rules

# %%
df[['antecedents','consequents']].value_counts().head(10)
# we get 116 sets appearing 8 rules appear in 2 seasons

# %%
print(finalTestRules[2].shape)
print(finalMainRules[2].shape)

# %% [markdown]
# ## Check:
# We want to verify below whether rules are statistically significant, in a season may be in another. season. We therefore look for overlap, in other season, for a the statistically significant rules of a tested season.

# %%
dfSeasonDiffTestStat, dfSeasonMainStat, dfSeasonOverlapStat = getDiffOverlap(finalMainRules,finalTestRules)
# we will compare the overlapping difference between those values


# %%
dfSeasonDiffTestStat[1] # statiscally siginificant bundles only in that season

# %%
print("Rules Significant in Spring not in another season:",dfSeasonDiffTestStat[0].shape[0])
print("Rules Significant in another season not in Spring:", dfSeasonMainStat[0].shape[0])
print("Rules Significant in Spring and in another season:",dfSeasonOverlapStat[0].shape[0])
RulesBehaviour =[]
RulesBehaviour.append([dfSeasonOverlapStat[0].shape[0],dfSeasonMainStat[0].shape[0],dfSeasonDiffTestStat[0].shape[0]])

# %%
print("Rules Significant in Summer not in another season:",dfSeasonDiffTestStat[1].shape[0])
print("Rules Significant in another season not in Summer:", dfSeasonMainStat[1].shape[0])
print("Rules Significant in Summer and in another season:",dfSeasonOverlapStat[1].shape[0])
RulesBehaviour.append([dfSeasonOverlapStat[1].shape[0],dfSeasonMainStat[1].shape[0],dfSeasonDiffTestStat[1].shape[0]])
RulesBehaviour.append([dfSeasonOverlapStat[2].shape[0],dfSeasonMainStat[2].shape[0],dfSeasonDiffTestStat[2].shape[0]])
RulesBehaviour.append([dfSeasonOverlapStat[3].shape[0],dfSeasonMainStat[3].shape[0],dfSeasonDiffTestStat[3].shape[0]])

# %% [markdown]
# # Part 7: Plotting the labels
# 
# We first make labels we will plot.

# %% [markdown]
# ### We can now check out the rules we obtained

# %%
dfSeasonMainStat[1] # statiscally siginificant bundles only in other season (may better or worse)

# %%
dfSeasonOverlapStat[3] # those are rules that were statistically significant in winter and another season

# %%
df= dfSeason_overlap[3] # wecheck the results for Winter
df

# %% [markdown]
# ## How to plot our results?
# 
# 

# %%
dfSeason_DiffInMain[2]

# %%
dfSeason_DiffInTest[2]

# %%
dfSeason_overlap[2]

# %%
statusOnlyInTest = 'Only in test'
StatisticallySignificantOverperform = 'Relative high confidence'
StatisticallySignificantUnderperform = 'Relative low confidence'
StatisticallySignificant = 'Statistically significant'


# %%

def assemblyClassification(dfSeason_DiffInTest,dfSeason_overlap):
    classification = []
    for season in range(seasons):
        DiffInTest = dfSeason_DiffInTest[season]
        DiffInTest['Classification'] = statusOnlyInTest
        df=  dfSeason_overlap[season][['antecedents','consequents','Chi-Square-Unbiased','Confidence Ratio']] 
        df['Classification'] = 'normal'
        # df[((df['Chi-Square'] >= ChiSquareCriticalValue90) & (df['Confidence Ratio'] >= 1.2)), 'Classification']=StatisticallySignificantOverperform
        df['Classification'] = np.where((df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue90), StatisticallySignificant, df['Classification'])
        df['Classification'] = np.where((df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue90) & (df['Confidence Ratio'] >= 1.5), StatisticallySignificantOverperform, df['Classification'])
        df['Classification'] = np.where((df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue90) & (df['Confidence Ratio'] <= 0.5), StatisticallySignificantUnderperform , df['Classification'])
        df = df.drop(columns=['Chi-Square-Unbiased','Confidence Ratio'])
        rulesTest = pd.concat([DiffInTest,df])
        rulesTest.reset_index(drop=True)
        classification.append(rulesTest)
        
    return classification


# %%
classification = assemblyClassification(dfSeason_DiffInTest,dfSeason_overlap)


# %%
dataFinal = []
for season in range(seasons):
    df = classification[season]
    countOnlyInTest = df['Classification'].loc[(df['Classification'] == statusOnlyInTest)].count()
    countNormal = df['Classification'].loc[(df['Classification'] == 'normal')].count()
    countUnder = df['Classification'].loc[(df['Classification'] == StatisticallySignificantUnderperform)].count()
    countSS = df['Classification'].loc[(df['Classification'] == StatisticallySignificant)].count()
    countOver = df['Classification'].loc[(df['Classification'] == StatisticallySignificantOverperform)].count()
    dataFinal.append([countOnlyInTest,countNormal,countUnder,countSS,countOver])
print(dataFinal)


# %%
dataFinal = np.array(dataFinal)
dataFinal = dataFinal.reshape(4,5)
dataFinal = np.transpose(dataFinal)
dataFinal

# %%


import numpy as np
import matplotlib.pyplot as plt
N = 4
InTestValues = dataFinal[0]
normalValues = dataFinal[1]
SSUnderValues = dataFinal[2]
SSValues = dataFinal[3]
SSOverValues = dataFinal[4]

ind = np.arange(N) # the x locations for the groups
width = 0.35
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ax.bar(ind, InTestValues, width, color='gray')
ax.bar(ind, normalValues, width,bottom=InTestValues, color='red') # you need to properly stack them ffs
ax.bar(ind, SSUnderValues, width,bottom=normalValues+InTestValues, color='lightblue')
ax.bar(ind, SSValues, width,bottom=SSUnderValues+normalValues+InTestValues, color='royalblue')
ax.bar(ind, SSOverValues, width,bottom=SSValues+SSUnderValues+normalValues+InTestValues, color='darkblue')
#ax.set_yticks(np.arange(0, 1000, 100))


for edge_i in ['top', 'right']:
    ax.spines[edge_i].set_edgecolor("white")

ax.set_ylabel('Count')
#ax.set_title('Scores by Season')
ax.set_xticks(ind, ('Spring', 'Summer', 'Fall', 'Winter'))
ax.set_yticks(np.arange(0, 1800, 200))
ax.legend(labels=["Only in the Tested Season",'Not Significant',"Low Confidence Significant","Normal Confidence Significant","High Confidence Significant"],loc='upper right', fontsize='x-small')
plt.show()

# %%
dataBarChart = np.copy(dataFinal)
print(dataBarChart)
dataBarChart  = np.delete(dataBarChart, 0, 0)
SSvalues = np.delete(dataBarChart, 0, 0)
SSvalues = np.transpose(SSvalues)
SSrow = np.sum(np.delete(dataBarChart, 0, 0),axis=0)
print(SSrow)
print(SSvalues)
normalValuesInOverlap = dataBarChart[0]

# %%
import numpy as np
import matplotlib.pyplot as plt
N = 4


ind = np.arange(N) # the x locations for the groups
width = 0.35
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(ind, normalValuesInOverlap, width, color='red')
ax.bar(ind, SSrow, width, color='blue')# bottom=menMeans
ax.set_ylabel('Count')
#ax.set_title('Normal anf statistically Significant values by season')

for edge_i in ['top', 'right']:
    ax.spines[edge_i].set_edgecolor("white")
    
ax.set_xticks(ind, ('Spring', 'Summer', 'Fall', 'Winter'))
ax.set_yticks(np.arange(0, 900, 200))
ax.legend(labels=['Not Significant', 'Statistically Significant'])
plt.show()

# %%
AllRules = rulesTestFold[3][['antecedents','consequents']] 
AllRules

# %% [markdown]
# ## Limitations
# 
# In the test fold, because we have less data, we obtain more rules than for a bigger dataset, as a particular rule needs less occurrence of its antecedents and consequents in order for the support to be above the threshold.
# 

# %%
       [[ 38,  10,   8,  60],
       [  9,   2,   4,  14],
       [ 25,   4,   4,  32]]

# %%
       [[ 34,   9,   8,  41],
       [ 18,   3,   4,  38],
       [ 20,   4,   4,  27]]

# %%
# Creating dataset
SSlabels = ['Low Confidence Ratio','Normal Confidence Ratio','High Confidence Ratio']
color = ["lightblue", "royalblue", "darkblue"]

for season in range(seasons):
    data = SSvalues[season]

    # Creating plot
    fig = plt.figure(figsize =(10, 7))
    plt.pie(data, labels = SSlabels, colors = color)

    # show plot
    plt.show()

# %%
range(seasons)

# %%
plt.figure(figsize=(15,5))
SSlabels = ['Low Confidence Ratio','Normal Confidence Ratio','High Confidence Ratio']
color = ["lightblue", "royalblue", "darkblue"]

plt.subplot(1,4,1)
plt.pie(SSvalues[0], colors = color, autopct='%1.2f%%', textprops={'color':"w", 'weight':'bold'} )
plt.title("Spring")
plt.subplot(1,4,2)
plt.pie(SSvalues[1], colors = color, autopct='%1.2f%%', textprops={'color':"w", 'weight':'bold'})
plt.title("Summer")
plt.subplot(1,4,3)
plt.pie(SSvalues[2], colors = color, autopct='%1.2f%%', textprops={'color':"w", 'weight':'bold'})
plt.title("Fall")
plt.subplot(1,4,4)
plt.pie(SSvalues[3] , colors = color, autopct='%1.2f%%', textprops={'color':"w", 'weight':'bold'})
plt.title("Winter")

plt.legend(labels = SSlabels, fontsize="small",loc='lower right' )


plt.show()

plt.savefig("fig.jpg")

# %%
# Creating dataset
Ruleslabels = ['Rules Significant in Overlap','Rules Significant - other Season','Rules Significant - only Season']
colorBehaviour = ["darkgreen", "lightgreen", "teal"]

for season in range(seasons):
    data = RulesBehaviour[season]

    # Creating plot
    fig = plt.figure(figsize =(10, 7))
    plt.pie(data, labels = Ruleslabels, colors = colorBehaviour)

    # show plot
    plt.show()

# %%
plt.figure(figsize=(15,5))


plt.subplot(1,4,1)
plt.pie(RulesBehaviour[0], colors = colorBehaviour, autopct='%1.2f%%', textprops={'color':"w", 'weight':'bold'} )
plt.title("Spring")
plt.subplot(1,4,2)
plt.pie(RulesBehaviour[1], colors = colorBehaviour, autopct='%1.2f%%', textprops={'color':"w", 'weight':'bold'})
plt.title("Summer")
plt.subplot(1,4,3)
plt.pie(RulesBehaviour[2], colors = colorBehaviour, autopct='%1.2f%%', textprops={'color':"w", 'weight':'bold'})
plt.title("Fall")
plt.subplot(1,4,4)
plt.pie(RulesBehaviour[3] , colors = colorBehaviour, autopct='%1.2f%%', textprops={'color':"w", 'weight':'bold'})
plt.title("Winter")

plt.legend(labels = Ruleslabels, fontsize="small",loc='lower right' )


plt.show()
plt.savefig("fig.jpg")

# %% [markdown]
# # Annex Code 
# We will consign here functions, that were used during our exploration of the data, but are not used above.

# %%
def getExpectationHomogeneous(listExpectedProbs,listObservedProbsMain):
        assert len(listObservedProbs) == len(listExpectedProbs)
        listHomogenous = []
        newValue = 0
        for value in range(len(listExpectedProbs)):
            observedTestLengthN = len(baskets[value].index)
            observedMainLengthN = fullBasketLength - observedTestLengthN
            newValue = ((listObservedProbs[value]+listExpectedProbs[value])*observedTestLengthN)/fullBasketLength
            listHomogenous.append(newValue)
        for value in range(len(listExpectedProbs)):
            observedTestLengthN = len(baskets[value].index)
            observedMainLengthN = fullBasketLength - observedTestLengthN
            newValue = ((listObservedProbs[value]+listExpectedProbs[value])*observedMainLengthN)/fullBasketLength
            listHomogenous.append(newValue)
            
        return listHomogenous

# %%
def ExpectedContingencyTableRule(n,support, lift, confidence):
     # we build a list 
    # containg the probabilitie of the Expected Contigency Table

    AandB = n*support/lift
    AandNotB = n*support/confidence*(1-confidence/lift)
    NotAandB = n*(1-support/confidence)*(confidence/lift)
    NotAandNotB = n*(1-support/confidence)*(1-confidence/lift)
    # those probabilities are computed as Specfied in the Boston College Paper
    return [AandB,AandNotB,NotAandB,NotAandNotB]

def ChiSquareComputationReloaded(n,support, lift, confidence):
    ChiSquareValue = n*np.power((lift-1),2)*(support*confidence)/((confidence-support)*(lift-confidence))
    # those are a different type of chi-Square to use on a set of rules
    # we may need it but it's not the most useful
    return ChiSquareValue

# %%
# chek the results for a antecedent
df = dfSeason_overlap[1].copy(deep=True)
df["antecedents"] = df["antecedents"].apply(lambda x: ''.join(list(x))).astype("unicode")
df.loc[(df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue90) & (df['antecedents'] == "whole milk")]

# %%
# chek the results for a consequent
df = dfSeason_overlap[3].copy(deep=True) # deep copy to not change our tables
df["consequents"] = df["consequents"].apply(lambda x: ''.join(list(x))).astype("unicode")
df.loc[(df['Chi-Square-Unbiased'] >= ChiSquareCriticalValue90) & (df['consequents'] == "whole milk")]


