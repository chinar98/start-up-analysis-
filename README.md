# startup-anaysis-
Jupyter Notebook
project fy
Last Checkpoint: 2 minutes ago
(autosaved)
Current Kernel Logo
Python 3 
File
Edit
View
Insert
Cell
Kernel
Widgets
Help

StartUp Funding
A startup with a brilliant business idea is aiming to get its operations up and running. From humble beginnings, the company proves the worthiness of its model and products, steadily growing thanks to the generosity of friends, family, and the founders' own financial resources. Over time, its customer base begins to grow, and the business begins to expand its operations and its aims. Before long, the company has risen through the ranks of its competitors to become highly valued, opening the possibilities for future expansion to include new offices, employees, and even an initial public offering (IPO)
This project shows the insights of funding done by startups and how growth changed with several factors. The aim of paper is to get a descriptive overview and a relationship pattern of funding and growth of newly launched startups. Another important point to understand how funding changes with time is an important aspect. Possible area of interests would be – (Funding ecosystem and time relation, cities as a important factor, which industries, important investors). Dataset we are using contains information of funding of startups from 1980 to 2014.The amount invested is in USD. Aggregation of data w.r.t cities, investors, funding type etc. is required to get an optimized result. Here we done major preprocessing of data and overcome problem of missing data and uncertain distributions. Also, Visualizations are done to find the anomalies and mining patterns from data. It seems to be some cities showing some abnormal behavior when it comes to funding

#Import some useful libraries
​
import pandas as pd
import seaborn as sns
import squarify
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn import preprocessing
​
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format
## Function for providing summary in dataframe
%matplotlib inline
​
def funding_information(data,name):
    company = data[data['name'] == name]
    print ("Company : ", name)
    print ("Total Funding : ", company.funding_total_usd.values[0] , " $")
    print ("Seed Funding : ", company.seed.values[0] , " $")
    print ("Angle Funding :", company.angel.values[0] , " $")
    print ("Grant Funding : ",company.grant.values[0] , " $")
    print ("Product Crowd Funding : ",company.product_crowdfunding.values[0] , " $")
    print ("Equity Crowd Funding : ",company.equity_crowdfunding.values[0] , " $")
    print ("Undisclode Funding : ", company.undisclosed.values[0] , " $")
    print ("Convertible Note : ", company.convertible_note.values[0] , " $")
    print ("Debt Financing : ", company.debt_financing.values[0] , " $")
    print ("Private Equity : ",company.private_equity.values[0] , " $")
    print ("PostIPO Equity : ",company.post_ipo_equity.values[0] , " $")
    print ("PostIPO Debt : ",company.post_ipo_debt.values[0] , " $")
    print ("Secondary Market : ",company.secondary_market.values[0] , " $")
    print ("Venture Funding : ",company.venture.values[0] , " $")
    print ("Round A funding : ",company.round_A.values[0] , " $")
    print ("Round B funding : ",company.round_B.values[0] , " $")
    print ("Round C funding : ",company.round_C.values[0] , " $")
    print ("Round D funding : ",company.round_D.values[0] , " $")
    print ("Round E funding : ",company.round_E.values[0] , " $")
    print ("Round F funding : ",company.round_F.values[0] , " $")
    print ("Round G funding : ",company.round_G.values[0] , " $")
    print ("Round H funding : ",company.round_H.values[0] , " $")
​
def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):        
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue        
        for s in [s for s in liste_keywords if s in liste]: 
            if pd.notnull(s): keyword_count[s] += 1
    #______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = [     ]
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count
​
​
def makeCloud(Dict,name,color):
    words = dict()
​
    for s in Dict:
        words[s[0]] = s[1]
​
        wordcloud = WordCloud(
                      width=1500,
                      height=750, 
                      background_color=color, 
                      max_words=50,
                      max_font_size=500, 
                      normalize_plurals=False)
        wordcloud.generate_from_frequencies(words)
​
​
    fig = plt.figure(figsize=(12, 8))
    plt.title(name)
    plt.imshow(wordcloud)
    plt.axis('off')
​
    plt.show()
data = pd.read_csv("D:/mit wpu/sem 3/ml mini project/investments.csv",encoding = "ISO-8859-1")
Loading data into data frame

data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 49438 entries, 0 to 49437
Data columns (total 39 columns):
permalink               49438 non-null object
name                    49437 non-null object
homepage_url            45989 non-null object
category_list           45477 non-null object
 market                 45470 non-null object
 funding_total_usd      49438 non-null object
status                  48124 non-null object
country_code            44165 non-null object
state_code              30161 non-null object
region                  44165 non-null object
city                    43322 non-null object
funding_rounds          49438 non-null int64
founded_at              38554 non-null object
founded_month           38482 non-null object
founded_quarter         38482 non-null object
founded_year            38482 non-null float64
first_funding_at        49438 non-null object
last_funding_at         49438 non-null object
seed                    49438 non-null int64
venture                 49438 non-null int64
equity_crowdfunding     49438 non-null int64
undisclosed             49438 non-null int64
convertible_note        49438 non-null int64
debt_financing          49438 non-null int64
angel                   49438 non-null int64
grant                   49438 non-null int64
private_equity          49438 non-null int64
post_ipo_equity         49438 non-null int64
post_ipo_debt           49438 non-null int64
secondary_market        49438 non-null int64
product_crowdfunding    49438 non-null object
round_A                 49438 non-null int64
round_B                 49438 non-null int64
round_C                 49438 non-null int64
round_D                 49438 non-null int64
round_E                 49438 non-null int64
round_F                 49438 non-null int64
round_G                 49438 non-null int64
round_H                 49438 non-null int64
dtypes: float64(1), int64(21), object(17)
memory usage: 14.7+ MB
data.head()
permalink	name	homepage_url	category_list	market	funding_total_usd	status	country_code	state_code	region	...	secondary_market	product_crowdfunding	round_A	round_B	round_C	round_D	round_E	round_F	round_G	round_H
0	/organization/waywire	#waywire	http://www.waywire.com	|Entertainment|Politics|Social Media|News|	News	17,50,000	acquired	USA	NY	New York City	...	0	0	0	0	0	0	0	0	0	0
1	/organization/tv-communications	&TV Communications	http://enjoyandtv.com	|Games|	Games	40,00,000	operating	USA	CA	Los Angeles	...	0	0	0	0	0	0	0	0	0	0
2	/organization/rock-your-paper	'Rock' Your Paper	http://www.rockyourpaper.org	|Publishing|Education|	Publishing	40,000	operating	EST	NaN	Tallinn	...	0	0	0	0	0	0	0	0	0	0
3	/organization/in-touch-network	(In)Touch Network	http://www.InTouchNetwork.com	|Electronics|Guides|Coffee|Restaurants|Music|i...	Electronics	15,00,000	operating	GBR	NaN	London	...	0	0	0	0	0	0	0	0	0	0
4	/organization/r-ranch-and-mine	-R- Ranch and Mine	NaN	|Tourism|Entertainment|Games|	Tourism	60,000	operating	USA	TX	Dallas	...	0	0	0	0	0	0	0	0	0	0
5 rows × 39 columns

len(data)
49438
data.tail()
permalink	name	homepage_url	category_list	market	funding_total_usd	status	country_code	state_code	region	...	secondary_market	product_crowdfunding	round_A	round_B	round_C	round_D	round_E	round_F	round_G	round_H
49433	/organization/zzish	Zzish	http://www.zzish.com	|Analytics|Gamification|Developer APIs|iOS|And...	Education	3,20,000	operating	GBR	NaN	London	...	0	0	0	0	0	0	0	0	0	0
49434	/organization/zznode-science-and-technology-co...	ZZNode Science and Technology	http://www.zznode.com	|Enterprise Software|	Enterprise Software	15,87,301	operating	CHN	NaN	Beijing	...	0	0	1587301	0	0	0	0	0	0	0
49435	/organization/zzzzapp-com	Zzzzapp Wireless ltd.	http://www.zzzzapp.com	|Web Development|Advertising|Wireless|Mobile|	Web Development	97,398	operating	HRV	NaN	Split	...	0	0	0	0	0	0	0	0	0	0
49436	/organization/a-list-games	[a]list games	http://www.alistgames.com	|Games|	Games	93,00,000	operating	NaN	NaN	NaN	...	0	0	0	0	0	0	0	0	0	0
49437	/organization/x	[x+1]	http://www.xplusone.com/	|Enterprise Software|	Enterprise Software	4,50,00,000	operating	USA	NY	New York City	...	0	0	16000000	10000000	0	0	0	0	0	0
5 rows × 39 columns

As you can see in two outputs above, we have 54,294 rows of data but some of them not contain any information. It may lead to misdirection summary when we do some analysis or visualize them.

Then, we just remove them by select only data which has "name" column

## select only data which name is not null
​
data = data[~data.name.isna()]
len(data)
49437
49437 we have around 49,437 companies left in our datase

print( data.columns.values )
['permalink' 'name' 'homepage_url' 'category_list' ' market '
 ' funding_total_usd ' 'status' 'country_code' 'state_code' 'region'
 'city' 'funding_rounds' 'founded_at' 'founded_month' 'founded_quarter'
 'founded_year' 'first_funding_at' 'last_funding_at' 'seed' 'venture'
 'equity_crowdfunding' 'undisclosed' 'convertible_note' 'debt_financing'
 'angel' 'grant' 'private_equity' 'post_ipo_equity' 'post_ipo_debt'
 'secondary_market' 'product_crowdfunding' 'round_A' 'round_B' 'round_C'
 'round_D' 'round_E' 'round_F' 'round_G' 'round_H']
some column name contains space in string, we decide to remove them first.

data.rename(columns={' funding_total_usd ': "funding_total_usd",
                    ' market ': "market"},inplace=True)
print("Frequency count of missing values")
data.apply(lambda X:sum(X.isnull())) 
#apply function is used to do mapping column-wise
#apply function can apply tranformations to each column individually
Frequency count of missing values
permalink                   0
name                        0
homepage_url             3449
category_list            3961
market                   3968
funding_total_usd           0
status                   1314
country_code             5272
state_code              19276
region                   5272
city                     6115
funding_rounds              0
founded_at              10884
founded_month           10956
founded_quarter         10956
founded_year            10956
first_funding_at            0
last_funding_at             0
seed                        0
venture                     0
equity_crowdfunding         0
undisclosed                 0
convertible_note            0
debt_financing              0
angel                       0
grant                       0
private_equity              0
post_ipo_equity             0
post_ipo_debt               0
secondary_market            0
product_crowdfunding        0
round_A                     0
round_B                     0
round_C                     0
round_D                     0
round_E                     0
round_F                     0
round_G                     0
round_H                     0
dtype: int64
plt.figure(figsize=(10,5)) #plt is the object of matplot lib and .figure() is used to show or change properties of graphs
sns.heatmap(data.isnull(),cmap='viridis',yticklabels=False,cbar=False)#heatmaps are matrix plots which can visualize data in 2D
plt.show()

data cleaning
print("Information of total number of non-empty columns")
print("-------------------------------------------------")
print(data.info(null_counts=True))
Information of total number of non-empty columns
-------------------------------------------------
<class 'pandas.core.frame.DataFrame'>
Int64Index: 49437 entries, 0 to 49437
Data columns (total 39 columns):
permalink               49437 non-null object
name                    49437 non-null object
homepage_url            45988 non-null object
category_list           45476 non-null object
market                  45469 non-null object
funding_total_usd       49437 non-null object
status                  48123 non-null object
country_code            44165 non-null object
state_code              30161 non-null object
region                  44165 non-null object
city                    43322 non-null object
funding_rounds          49437 non-null int64
founded_at              38553 non-null object
founded_month           38481 non-null object
founded_quarter         38481 non-null object
founded_year            38481 non-null float64
first_funding_at        49437 non-null object
last_funding_at         49437 non-null object
seed                    49437 non-null int64
venture                 49437 non-null int64
equity_crowdfunding     49437 non-null int64
undisclosed             49437 non-null int64
convertible_note        49437 non-null int64
debt_financing          49437 non-null int64
angel                   49437 non-null int64
grant                   49437 non-null int64
private_equity          49437 non-null int64
post_ipo_equity         49437 non-null int64
post_ipo_debt           49437 non-null int64
secondary_market        49437 non-null int64
product_crowdfunding    49437 non-null object
round_A                 49437 non-null int64
round_B                 49437 non-null int64
round_C                 49437 non-null int64
round_D                 49437 non-null int64
round_E                 49437 non-null int64
round_F                 49437 non-null int64
round_G                 49437 non-null int64
round_H                 49437 non-null int64
dtypes: float64(1), int64(21), object(17)
memory usage: 15.1+ MB
None
Next, we will delete conditions with only zero startup

df_condition = data.groupby(['city'])['market'].nunique().sort_values(ascending=False)
df_condition = pd.DataFrame(df_condition).reset_index()
df_condition.tail(20)
city	market
4168	Ferrières	0
4169	Fort Dodge	0
4170	France	0
4171	Frontenac	0
4172	Frunze	0
4173	Garching Bei München	0
4174	Garland	0
4175	Glenrothes	0
4176	Goes	0
4177	Gometz-la-ville	0
4178	Grady	0
4179	Guaporé	0
4180	Guwahati	0
4181	Gwinn	0
4182	Hagerman	0
4183	Hammel	0
4184	Hector	0
4185	Heiloo	0
4186	Hellerup	0
4187	's-hertogenbosch	0
print("Frequency count of missing values")
data.apply(lambda X:sum(X.isnull())) 
Frequency count of missing values
permalink                   0
name                        0
homepage_url             3449
category_list            3961
market                   3968
funding_total_usd           0
status                   1314
country_code             5272
state_code              19276
region                   5272
city                     6115
funding_rounds              0
founded_at              10884
founded_month           10956
founded_quarter         10956
founded_year            10956
first_funding_at            0
last_funding_at             0
seed                        0
venture                     0
equity_crowdfunding         0
undisclosed                 0
convertible_note            0
debt_financing              0
angel                       0
grant                       0
private_equity              0
post_ipo_equity             0
post_ipo_debt               0
secondary_market            0
product_crowdfunding        0
round_A                     0
round_B                     0
round_C                     0
round_D                     0
round_E                     0
round_F                     0
round_G                     0
round_H                     0
dtype: int64
percent = (data.isnull().sum()).sort_values(ascending=False)
percent.plot(kind="bar", figsize = (14,6), fontsize = 10, color='steelblue')
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("", fontsize = 20)
plt.title("Total Missing Value ", fontsize = 20)
Text(0.5, 1.0, 'Total Missing Value ')

print(data.describe())
       funding_rounds  founded_year         seed       venture  \
count        49437.00      38481.00     49437.00      49437.00   
mean             1.70       2007.36    217325.39    7501202.27   
std              1.29          7.58   1056995.16   28471392.13   
min              1.00       1902.00         0.00          0.00   
25%              1.00       2006.00         0.00          0.00   
50%              1.00       2010.00         0.00          0.00   
75%              2.00       2012.00     25000.00    5000000.00   
max             18.00       2014.00 130000000.00 2351000000.00   

       equity_crowdfunding  undisclosed  convertible_note  debt_financing  \
count             49437.00     49437.00          49437.00        49437.00   
mean               6163.45    130223.92          23364.57      1888195.09   
std              199906.84   2981433.75        1432060.21    138205963.71   
min                   0.00         0.00              0.00            0.00   
25%                   0.00         0.00              0.00            0.00   
50%                   0.00         0.00              0.00            0.00   
75%                   0.00         0.00              0.00            0.00   
max            25000000.00 292432833.00      300000000.00  30079503000.00   

            angel        grant  ...  post_ipo_debt  secondary_market  \
count    49437.00     49437.00  ...       49437.00          49437.00   
mean     65420.30    162848.57  ...      443444.94          38456.70   
std     658297.38   5612144.71  ...    34282036.20        3864499.70   
min          0.00         0.00  ...           0.00              0.00   
25%          0.00         0.00  ...           0.00              0.00   
50%          0.00         0.00  ...           0.00              0.00   
75%          0.00         0.00  ...           0.00              0.00   
max   63590263.00 750500000.00  ...  5800000000.00      680611554.00   

           round_A      round_B      round_C       round_D      round_E  \
count     49437.00     49437.00     49437.00      49437.00     49437.00   
mean    1243980.18   1492921.35   1205380.18     737540.98    342475.13   
std     5532027.16   7472777.02   7993670.75    9815316.90   5406969.03   
min           0.00         0.00         0.00          0.00         0.00   
25%           0.00         0.00         0.00          0.00         0.00   
50%           0.00         0.00         0.00          0.00         0.00   
75%           0.00         0.00         0.00          0.00         0.00   
max   319000000.00 542000000.00 490000000.00 1200000000.00 400000000.00   

            round_F       round_G      round_H  
count      49437.00      49437.00     49437.00  
mean      169772.62      57671.83     14232.26  
std      6277968.90    5252365.05   2716892.77  
min            0.00          0.00         0.00  
25%            0.00          0.00         0.00  
50%            0.00          0.00         0.00  
75%            0.00          0.00         0.00  
max   1060000000.00 1000000000.00 600000000.00  

[8 rows x 22 columns]
data['funding_total_usd'].describe()
count     49437
unique    14617
top        -   
freq       8531
Name: funding_total_usd, dtype: object
data["funding_total_usd"].replace(15912914, np.nan, inplace= True)
#replacing with mean
data.fillna(method='bfill')
permalink	name	homepage_url	category_list	market	funding_total_usd	status	country_code	state_code	region	...	secondary_market	product_crowdfunding	round_A	round_B	round_C	round_D	round_E	round_F	round_G	round_H
0	/organization/waywire	#waywire	http://www.waywire.com	|Entertainment|Politics|Social Media|News|	News	17,50,000	acquired	USA	NY	New York City	...	0	0	0	0	0	0	0	0	0	0
1	/organization/tv-communications	&TV Communications	http://enjoyandtv.com	|Games|	Games	40,00,000	operating	USA	CA	Los Angeles	...	0	0	0	0	0	0	0	0	0	0
2	/organization/rock-your-paper	'Rock' Your Paper	http://www.rockyourpaper.org	|Publishing|Education|	Publishing	40,000	operating	EST	TX	Tallinn	...	0	0	0	0	0	0	0	0	0	0
3	/organization/in-touch-network	(In)Touch Network	http://www.InTouchNetwork.com	|Electronics|Guides|Coffee|Restaurants|Music|i...	Electronics	15,00,000	operating	GBR	TX	London	...	0	0	0	0	0	0	0	0	0	0
4	/organization/r-ranch-and-mine	-R- Ranch and Mine	http://nic.club/	|Tourism|Entertainment|Games|	Tourism	60,000	operating	USA	TX	Dallas	...	0	0	0	0	0	0	0	0	0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
49433	/organization/zzish	Zzish	http://www.zzish.com	|Analytics|Gamification|Developer APIs|iOS|And...	Education	3,20,000	operating	GBR	NY	London	...	0	0	0	0	0	0	0	0	0	0
49434	/organization/zznode-science-and-technology-co...	ZZNode Science and Technology	http://www.zznode.com	|Enterprise Software|	Enterprise Software	15,87,301	operating	CHN	NY	Beijing	...	0	0	1587301	0	0	0	0	0	0	0
49435	/organization/zzzzapp-com	Zzzzapp Wireless ltd.	http://www.zzzzapp.com	|Web Development|Advertising|Wireless|Mobile|	Web Development	97,398	operating	HRV	NY	Split	...	0	0	0	0	0	0	0	0	0	0
49436	/organization/a-list-games	[a]list games	http://www.alistgames.com	|Games|	Games	93,00,000	operating	USA	NY	New York City	...	0	0	0	0	0	0	0	0	0	0
49437	/organization/x	[x+1]	http://www.xplusone.com/	|Enterprise Software|	Enterprise Software	4,50,00,000	operating	USA	NY	New York City	...	0	0	16000000	10000000	0	0	0	0	0	0
49437 rows × 39 columns

EDA
lets understand what is there in this data

plt.rcParams['figure.figsize'] = 10,10
labels = data['status'].value_counts().index.tolist()
sizes = data['status'].value_counts().tolist()
explode = (0, 0.050,0.3)
colors = ['whitesmoke','coral','red']
​
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=20)
plt.axis('equal')
plt.tight_layout()
plt.title("What is start up companies current status", fontdict=None, position= [0.48,1], size = 'x-large')
plt.show()

Most of company (86.9 %) in this dataset is operating, and around 5.4 % company is already closed.

data['status'].value_counts()
operating    41829
acquired      3692
closed        2602
Name: status, dtype: int64
len(data['market'].unique())
754
data['market'].value_counts()[:5]
 Software          4620
 Biotechnology     3688
 Mobile            1983
 E-Commerce        1805
 Curated Web       1655
Name: market, dtype: int64
because we have around 754 categories of start up, Then just plot the top 15 : )

plt.rcParams['figure.figsize'] = 15,8
​
height = data['market'].value_counts()[:15].tolist()
bars =  data['market'].value_counts()[:15].index.tolist()
y_pos = np.arange(len(bars))
plt.bar(y_pos, height , width=0.7 ,color= ['c']+['teal']*14)
plt.xticks(y_pos, bars)
plt.xticks(rotation=90)
plt.title("Top 15 Start-Up market category", fontdict=None, position= [0.48,1.05], size = 'x-large')
plt.show()

set_keywords = set()
for liste_keywords in data['category_list'].str.split('|').values:
    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
#_________________________
# remove null chain entry
set_keywords.remove('')
keyword_occurences, dum = count_word(data, 'category_list', set_keywords)
​
makeCloud(keyword_occurences[0:15],"Keywords","Whitesmoke")

df_new = data[data['founded_year'] >= 2000]
df_new['founded_year'] = df_new['founded_year'].astype(int)
plt.figure(figsize = (16,7))
sns.countplot(x = 'founded_year', data = df_new)
plt.show()

the above fig shows year wise startup has been founded

now lets look at the countries in which startup has been founded

plt.figure(figsize=(16,7))
g = sns.countplot(x ='country_code', data = data, order=data['country_code'].value_counts().iloc[:10].index)
plt.xticks(rotation=30)
plt.show()

we can see that USA has more number of startup

df_USA = data[(data['country_code'] =='USA')]
plt.figure(figsize=(16,7))
g = sns.countplot(x ='state_code', data = df_USA, order=data['state_code'].value_counts().iloc[:10].index)
plt.xticks(rotation=30)
plt.show()

city_market = data.groupby(['city'])['market'].nunique().sort_values(ascending=False)
city_market[0:20].plot(kind="bar", figsize = (14,6), fontsize = 10,color="steelblue")
plt.xlabel("", fontsize = 20)
plt.ylabel("", fontsize = 20)
plt.title("Top20 : The number of of startups in a city", fontsize = 20)
Text(0.5, 1.0, 'Top20 : The number of of startups in a city')

Country_code with MarkeT
data['count'] = 1
country_market = data[['count','country_code','market']].groupby(['country_code','market']).agg({'count': 'sum'})
# Change: groupby state_office and divide by sum
country_market_pct = country_market.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))
country_market_pct.reset_index(inplace = True)
I want to know the different of startup market between USA and our country

USA_market_pct = country_market_pct[country_market_pct['country_code'] == "USA"]
USA_market_pct = USA_market_pct.sort_values('count',ascending = False)[0:10]
​
​
## USA
plt.rcParams['figure.figsize'] =10,10
labels = list(USA_market_pct['market'])+['Other...']
sizes = list(USA_market_pct['count'])+[100-USA_market_pct['count'].sum()]
explode = (0.18, 0.12, 0.09,0,0,0,0,0,0,0,0.01)
colors =  ['lightsteelblue','mediumaquamarine','moccasin'] +['khaki']*8
​
plt.pie(sizes, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=30)
plt.axis('equal')
plt.tight_layout()
plt.title("USA start up market", fontdict=None, position= [0.48,1.1], size = 'x-large')
​
plt.show()

For USA, Most of start up market is about Software & Technology

IND_market_pct = country_market_pct[country_market_pct['country_code'] == "IND"]
IND_market_pct = IND_market_pct.sort_values('count',ascending = False)[0:10]
plt.rcParams['figure.figsize'] = 10,10
labels = list(IND_market_pct['market'])+['Other...']
sizes = list(IND_market_pct['count'])+[100-USA_market_pct['count'].sum()]
explode = (0.18, 0.12, 0.09,0,0,0,0,0,0,0,0.01)
colors =  ['skyblue','violet','gold'] +['wheat']*8
​
plt.pie(sizes, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=30)
plt.axis('equal')
plt.tight_layout()
plt.title("Indian start up market", fontdict=None, position= [0.48,1.1], size = 'x-large')
plt.show()
​

For our country(India), The most popular market is E-commerce but other category is about Shopping, Marketplace, Social Network . . . these two pie charts show how different of interest trend between INDIA and America

Total Funding USD
data['funding_total_usd'].head()
0    17,50,000
1    40,00,000
2       40,000
3    15,00,000
4       60,000
Name: funding_total_usd, dtype: object
data['funding_total_usd'] = data['funding_total_usd'].str.replace(',', '')
data['funding_total_usd'] = data['funding_total_usd'].str.replace('-', '')
data['funding_total_usd'] = data['funding_total_usd'].str.replace(' ', '')
​
data['funding_total_usd'] = pd.to_numeric(data['funding_total_usd'], errors='coerce')
​
data['funding_total_usd'].head()
0   1750000.00
1   4000000.00
2     40000.00
3   1500000.00
4     60000.00
Name: funding_total_usd, dtype: float64
plt.rcParams['figure.figsize'] = 15,6
plt.hist(data['funding_total_usd'].dropna(), bins=30)
plt.ylabel('Count')
plt.xlabel('Fnding (usd)')
plt.title("Distribution of total funding ")
plt.show()

Seem like it has large gap between the highest value and the lowest, let ignore outlier first we will use the simple remove outlier technique such as 1.5IQR

Multiply the interquartile range (IQR) by 1.5 (a constant used to discern outliers). Add 1.5 x (IQR) to the third quartile. Any number greater than this is a suspected outlier. Subtract 1.5 x (IQR) from the first quartile. Any number less than this is a suspected outlier

Q1 = data['funding_total_usd'].quantile(0.25)
Q3 = data['funding_total_usd'].quantile(0.75)
IQR = Q3 - Q1
​
lower_bound = (Q1 - 1.5 * IQR)
upper_bound = (Q3 + 1.5 * IQR)
without_outlier = data[(data['funding_total_usd'] > lower_bound ) & (data['funding_total_usd'] < upper_bound)]
​
​
plt.rcParams['figure.figsize'] = 15,6
plt.hist(without_outlier['funding_total_usd'].dropna(), bins=30,color = 'khaki' )
​
plt.ylabel('Count')
plt.xlabel('Funding (usd)')
plt.title("Distribution of total funding ", fontdict=None, position= [0.48,1.05], size = 'x-large')
plt.show()

The average funding is around 15 m usd.

Let see the position of well-known company like Facebook, Alibaba, Uber but those companies are unicorn startup!!. Then, we need to move hist plot to focus only funding >= 1 billion usd

Facebook_total_funding = data['funding_total_usd'][data['name']=="Facebook"].values[0]
Uber_total_funding = data['funding_total_usd'][data['name']=="Uber"].values[0]
Alibaba_total_funding = data['funding_total_usd'][data['name']=="Alibaba"].values[0]
Cloudera_total_funding = data['funding_total_usd'][data['name']=="Cloudera"].values[0]
Flipkart_total_funding = data['funding_total_usd'][data['name']=="Flipkart"].values[0]
​
​
plt.rcParams['figure.figsize'] = 15,6
​
plt.hist(data['funding_total_usd'][(data['funding_total_usd'] >= 1000000000)&(data['funding_total_usd'] <= 3000000000)].dropna(), bins=30,color = 'silver' )
plt.ylabel('Count')
plt.xlabel('Funding (usd)')
plt.title("Where are the well-known companies ? ", fontdict=None, position= [0.48,1.05], size = 'x-large')
​
plt.axvline(Facebook_total_funding,color='red',linestyle ="--")
plt.text(Facebook_total_funding+15000000, 2.6,"Flipkar")
#plt.text(Facebook_total_funding+15000000, 2.6,"Flipkar")TO PRINT FLIPKART
​
plt.axvline(Uber_total_funding,color='black',linestyle ="--")
plt.text(Uber_total_funding+10000000, 2.2,"Uber")
​
plt.axvline(Cloudera_total_funding,color='blue',linestyle ="--")
plt.text(Cloudera_total_funding+10000000, 1.9,"Cloudera")
​
plt.axvline(Alibaba_total_funding,color='orange',linestyle ="--")
plt.text(Alibaba_total_funding+10000000, 1.6,"Alibaba")
plt.ticklabel_format(style='plain')
​
​
​
plt.show()
​

But..., Are they the highest funding ? the answer is no

Verizon_total_funding = data['funding_total_usd'][data['name']=="Verizon Communications"].values[0]
Sberbank_total_funding = data['funding_total_usd'][data['name']=="Sberbank"].values[0]
​
plt.rcParams['figure.figsize'] = 15,6
plt.hist(data['funding_total_usd'][(data['funding_total_usd'] >= 1000000000)].dropna(), bins=30,color = 'gold' )
plt.ylabel('Count')
plt.xlabel('Funding (usd)')
plt.title("Who get the highest funding ? ", fontdict=None, position= [0.48,1.05], size = 'x-large')
​
plt.axvline(Facebook_total_funding,color='royalblue',linestyle ="--")
plt.text(Facebook_total_funding+15000000, 11,"Facebook")
​
plt.axvline(Uber_total_funding,color='pink',linestyle ="--")
plt.text(Uber_total_funding+10000000, 9,"Uber")
​
plt.axvline(Cloudera_total_funding,color='dodgerblue',linestyle ="--")
plt.text(Cloudera_total_funding+10000000, 7,"Cloudera")
​
plt.axvline(Alibaba_total_funding,color='k',linestyle ="--")
plt.text(Alibaba_total_funding+10000000, 4,"Alibaba")
​
plt.axvline(Verizon_total_funding,color='red',linestyle ="--")
plt.text(Verizon_total_funding+100000000, 15,"Verizon Communications")
​
plt.axvline(Sberbank_total_funding,color='mediumseagreen',linestyle ="--")
plt.text(Sberbank_total_funding+100000000, 12,"Sberbank")
​
plt.show()

The most funding company in this dataset is "Verizon communication" which has total fund around 30,000,000,000 usd

Found at
data['founded_at'].head()
0    01-06-2012
1           NaN
2    26-10-2012
3    01-04-2011
4    01-01-2014
Name: founded_at, dtype: object
This column is provided in term of string format which we need to convert to datetime first

data['founded_at'] = pd.to_datetime(data['founded_at'], errors = 'coerce' )
plt.rcParams['figure.figsize'] = 15,6
data['name'].groupby(data["founded_at"].dt.year).count().plot(kind="line")
​
plt.ylabel('Count')
plt.title("Founded distribution ", fontdict=None, position= [0.48,1.05], size = 'x-large')
plt.show()

Facebook_founded_year = data['founded_at'][data['name']=="Facebook"].dt.year.values[0]
Uber_founded_year  = data['founded_at'][data['name']=="Uber"].dt.year.values[0]
Alibaba_founded_year  = data['founded_at'][data['name']=="Alibaba"].dt.year.values[0]
Flipkart_founded_year=data['founded_at'][data['name']=="Flipkart"].dt.year.values[0]
Uber_founded_year
2009
Flipkart_founded_year
​
2007
Alibaba_founded_year
1999
plt.rcParams['figure.figsize'] = 15,6
data['name'][data["founded_at"].dt.year >= 1990].groupby(data["founded_at"].dt.year).count().plot(kind="line")
plt.ylabel('Count')
​
plt.axvline(Facebook_founded_year,color='royalblue',linestyle ="--")
plt.text(Facebook_founded_year+0.15, 3000,"Facebook \n (2004)")
​
plt.axvline(Uber_founded_year,color='black',linestyle ="--")
plt.text(Uber_founded_year+0.15, 4000,"Uber \n(2009)")
​
plt.axvline(Alibaba_founded_year,color='orange',linestyle ="--")
plt.text(Alibaba_founded_year+0.15, 2000,"Alibaba \n(1999)")
​
plt.axvline(Flipkart_founded_year,color='teal',linestyle ="--")
plt.text(Flipkart_founded_year+0.15, 2000,"Flipkat \n(2007)")
​
plt.title("When the well-known company found ?", fontdict=None, position= [0.48,1.05], size = 'x-large')
plt.show()

Funding Analysis
funding_information(data,"Dropbox")
Company :  Dropbox
Total Funding :  1107215000.0  $
Seed Funding :  15000  $
Angle Funding : 0  $
Grant Funding :  0  $
Product Crowd Funding :  0  $
Equity Crowd Funding :  0  $
Undisclode Funding :  0  $
Convertible Note :  0  $
Debt Financing :  500000000  $
Private Equity :  0  $
PostIPO Equity :  0  $
PostIPO Debt :  0  $
Secondary Market :  0  $
Venture Funding :  607200000  $
Round A funding :  7200000  $
Round B funding :  250000000  $
Round C funding :  350000000  $
Round D funding :  0  $
Round E funding :  0  $
Round F funding :  0  $
Round G funding :  0  $
Round H funding :  0  $
funding_information(data,"Flipkart")
Company :  Flipkart
Total Funding :  2351140000.0  $
Seed Funding :  0  $
Angle Funding : 140000  $
Grant Funding :  0  $
Product Crowd Funding :  0  $
Equity Crowd Funding :  0  $
Undisclode Funding :  0  $
Convertible Note :  0  $
Debt Financing :  0  $
Private Equity :  0  $
PostIPO Equity :  0  $
PostIPO Debt :  0  $
Secondary Market :  0  $
Venture Funding :  2351000000  $
Round A funding :  1000000  $
Round B funding :  10000000  $
Round C funding :  20000000  $
Round D funding :  150000000  $
Round E funding :  360000000  $
Round F funding :  210000000  $
Round G funding :  1000000000  $
Round H funding :  600000000  $
funding_information(data,"Facebook")
Company :  Facebook
Total Funding :  2425700000.0  $
Seed Funding :  0  $
Angle Funding : 500000  $
Grant Funding :  0  $
Product Crowd Funding :  0  $
Equity Crowd Funding :  0  $
Undisclode Funding :  0  $
Convertible Note :  0  $
Debt Financing :  100000000  $
Private Equity :  1710000000  $
PostIPO Equity :  0  $
PostIPO Debt :  0  $
Secondary Market :  0  $
Venture Funding :  615200000  $
Round A funding :  12700000  $
Round B funding :  27500000  $
Round C funding :  375000000  $
Round D funding :  200000000  $
Round E funding :  0  $
Round F funding :  0  $
Round G funding :  0  $
Round H funding :  0  $
Seed funding
Seed funding is the first official equity funding stage. It typically represents the first official money that a business venture or enterprise raises; some companies never extend beyond seed funding into Series A rounds or beyond.

There are other types of funding rounds available to startups, depending upon the industry and the level of interest among potential investors. It's not uncommon for startups to engage in what is known as "seed" funding or angel investor funding at the outset. Next, these funding rounds can be followed by Series A, B, and C funding rounds, as well as additional efforts to earn capital as well, if appropriate. Series A, B, and C are necessary ingredients for a business that decides “bootstrapping,” or merely surviving off of the generosity of friends, family, and the depth of their own pockets, will not suffice.

data[['name','seed']].head(5)
name	seed
0	#waywire	1750000
1	&TV Communications	0
2	'Rock' Your Paper	40000
3	(In)Touch Network	1500000
4	-R- Ranch and Mine	0
Average funding in this stage ? Note we need to beware when use the mean value Most of value in this column is 0, they will drag your average value down The solution is using data which is not 0 to find average

print("The average of seed funding stage is around ",data['seed'][data['seed'] != 0].mean(), "$")
The average of seed funding stage is around  776350.5418021533 $
How many company get funding in seed stage ?

data['get_funding_in_seed'] = data['seed'].map(lambda s :1  if s > 0 else 0)
plt.rcParams['figure.figsize'] =10,10
labels = ['cannot get funding','Get funding']
sizes = data['get_funding_in_seed'].value_counts().tolist()
explode = (0, 0.1)
colors =  ['skyblue','steelblue'] 
​
plt.pie(sizes, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=190)
plt.axis('equal')
plt.tight_layout()
plt.title("How may company get funding in seed stage", fontdict=None, position= [0.48,1.1], size = 'x-large')
​
plt.show()

## Remove Outlier first 
​
Q1 = data['seed'][data['seed'] != 0].quantile(0.25)
Q3 = data['seed'][data['seed'] != 0].quantile(0.75)
IQR = Q3 - Q1
​
lower_bound = (Q1 - 1.5 * IQR)
upper_bound = (Q3 + 1.5 * IQR)
without_outlier = data[(data['seed'] > lower_bound ) & (data['seed'] < upper_bound)]
​
​
Facebook_seed_funding = data['seed'][data['name']=="Facebook"].values[0]
Uber_seed_funding   = data['seed'][data['name']=="Uber"].values[0]
Dropbox_seed_funding   = data['seed'][data['name']=="Dropbox"].values[0]
Flipkart_seed_funding   = data['seed'][data['name']=="Flipkart"].values[0]
Netflix_seed_funding   = data['seed'][data['name']=="Netflix"].values[0]
Alibaba_seed_funding   = data['seed'][data['name']=="Alibaba"].values[0]
​
plt.rcParams['figure.figsize'] = 15,6
plt.hist(without_outlier['seed'][without_outlier['seed']!=0].dropna(), bins=50,color = 'khaki' )
​
plt.axvline(Facebook_seed_funding,color='blue',linestyle ="--")
plt.text(Facebook_seed_funding+0.15, 200,"Facebook \n ( 0$ )")
​
plt.axvline(Uber_seed_funding,color='black',linestyle ="--")
plt.text(Uber_seed_funding+0.15, 2000,"Uber \n ( 200000$ )")
​
plt.axvline(Dropbox_seed_funding,color='coral',linestyle ="--")
plt.text(Dropbox_seed_funding+0.15, 1000,"  Dropbox \n( 15000$ )")
​
​
​
​
plt.ylabel('Count')
plt.xlabel('Funding (usd)')
plt.title("Distribution of Seed funding ", fontdict=None, position= [0.48,1.05], size = 'x-large')
plt.show()

Angel funding
Who is angel?

An angel investor (also known as a private investor, seed investor or angel funder) is a high net worth individual who provides financial backing for small startups or entrepreneurs, typically in exchange for ownership equity in the company. Often, angel investors are found among an entrepreneur's family and friends. The funds that angel investors provide may be a one-time investment to help the business get off the ground or an ongoing injection to support and carry the company through its difficult early stages.

print("The average of Angel funding is around ",data['angel'][data['angel'] != 0].mean(), "$")
The average of Angel funding is around  1033615.6954298498 $
data['get_funding_in_angel'] = data['angel'].map(lambda s :"Get funding"  if s > 0 else "Not get funding")
​
​
​
print("Only " , data['get_funding_in_angel'].value_counts().values[1], " companies has angel investor")
print("while " , data['get_funding_in_angel'].value_counts().values[0], " are not")
print("~",data['get_funding_in_angel'].value_counts().values[1]/(data['get_funding_in_angel'].value_counts().values[1]
+data['get_funding_in_angel'].value_counts().values[0]) *100, "percent")
Only  3129  companies has angel investor
while  46308  are not
~ 6.329267552642757 percent
Investment in each round
Depending on the type of industry and investors, a funding round can take anywhere from three months to over a year. The time between each round can vary between six months to one year. Funds are offered by investors, usually angel investors or venture capital firms, which then receive a stake in the startup

data['round_A'][data['round_A'] != 0].mean()
6830906.178162835
data['round_B'][data['round_B'] != 0].mean()
13549761.864145402
data['round_C'][data['round_C'] != 0].mean()
21004716.314416636
data['round_D'][data['round_D'] != 0].mean()
28308861.312111802
data['round_E'][data['round_E'] != 0].mean()
32811904.83914729
data['round_F'][data['round_F'] != 0].mean()
48796797.72674418
data['round_G'][data['round_G'] != 0].mean()
83856543.3235294
data['round_H'][data['round_H'] != 0].mean()
175900000.0
data['round_H'][data['round_H'] != 0].mean()
175900000.0
round = ['round_A','round_B','round_C','round_D','round_E','round_F','round_G','round_H']
amount = [data['round_A'][data['round_A'] != 0].mean(),
          data['round_B'][data['round_B'] != 0].mean(),
          data['round_C'][data['round_C'] != 0].mean(),
          data['round_D'][data['round_D'] != 0].mean(),
          data['round_E'][data['round_E'] != 0].mean(),
          data['round_F'][data['round_F'] != 0].mean(),
          data['round_G'][data['round_G'] != 0].mean(),
         data['round_H'][data['round_H'] != 0].mean()]
​
​
​
plt.rcParams['figure.figsize'] = 15,8
​
height = amount
bars =  round
y_pos = np.arange(len(bars))
plt.bar(y_pos, height , width=0.7, color= ['cornsilk','oldlace','papayawhip','wheat','moccasin','navajowhite','burlywood','goldenrod'] )
plt.xticks(y_pos, bars)
plt.title("Average investment in each round", fontdict=None, position= [0.48,1.05], size = 'x-large')
plt.show()

import squarify
plt.figure(figsize=(17,12))
mean_amount = data.groupby("funding_rounds").mean()["funding_total_usd"].astype('int').sort_values(ascending=False).iloc[1:].head(15)
squarify.plot(sizes=mean_amount.values,label=mean_amount.index, value=mean_amount.values,color=['crimson','seagreen','olive','hotpink','deepskyblue','grey','purple','lime','yellow','orange'])
plt.title('Distribution of Startups across Top cities')
Text(0.5, 1.0, 'Distribution of Startups across Top cities')

total funding and founding round heat map

#The average funding from seed and angel funding in india by region
​
gbf=data[(data['country_code'] == 'IND')]
rg= gbf.groupby('region').mean()
fr=rg.plot(kind ='line', y=['seed','angel'], figsize=(15,5))
​
fr.set_title('Startups\'s,seed and angel in  india by region',fontsize=(20))
Text(0.5, 1.0, "Startups's,seed and angel in  india by region")

#The average funding from seed and angel in india by market
​
gbf=data[(data['country_code'] == 'IND')]
rg= gbf.groupby('market').mean()
fr=rg.plot(kind ='line', y=['seed','angel'], figsize=(15,5))
​
fr.set_title('Startups\'s,e funding from seed and angel in india by market in  india by market',fontsize=(20))
Text(0.5, 1.0, "Startups's,e funding from seed and angel in india by market in  india by market")

Time series
A time series is a series of data points indexed in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data.

x=data[["founded_year", "seed","venture"]]
x.head()
founded_year	seed	venture
0	2012.00	1750000	0
1	nan	0	4000000
2	2012.00	40000	0
3	2011.00	1500000	0
4	2014.00	0	0
x.plot()
plt.show()

data['founded_at'] = pd.to_datetime(data['founded_at'], errors = 'coerce' )
data['seed'].groupby(data["founded_at"].dt.year).count().plot(kind="line")
<matplotlib.axes._subplots.AxesSubplot at 0x1eb23599ba8>

data['seed_YN'] = ['yes' if x>0 else 'no' for x in data['seed']]
data['venture_YN'] = ['yes' if x>0 else 'no' for x in data['venture']]
g=data.groupby('seed_YN')
print(g.groups)
group_seed=pd.DataFrame(g.get_group('yes'))
group_seed
{'no': Int64Index([    1,     4,     5,     6,     7,     8,    11,    12,    13,
               14,
            ...
            49423, 49424, 49425, 49426, 49427, 49429, 49430, 49432, 49434,
            49437],
           dtype='int64', length=35598), 'yes': Int64Index([    0,     2,     3,     9,    10,    15,    17,    18,    20,
               22,
            ...
            49396, 49398, 49400, 49407, 49413, 49428, 49431, 49433, 49435,
            49436],
           dtype='int64', length=13839)}
permalink	name	homepage_url	category_list	market	funding_total_usd	status	country_code	state_code	region	...	round_D	round_E	round_F	round_G	round_H	count	get_funding_in_seed	get_funding_in_angel	seed_YN	venture_YN
0	/organization/waywire	#waywire	http://www.waywire.com	|Entertainment|Politics|Social Media|News|	News	1750000.00	acquired	USA	NY	New York City	...	0	0	0	0	0	1	1	Not get funding	yes	no
2	/organization/rock-your-paper	'Rock' Your Paper	http://www.rockyourpaper.org	|Publishing|Education|	Publishing	40000.00	operating	EST	NaN	Tallinn	...	0	0	0	0	0	1	1	Not get funding	yes	no
3	/organization/in-touch-network	(In)Touch Network	http://www.InTouchNetwork.com	|Electronics|Guides|Coffee|Restaurants|Music|i...	Electronics	1500000.00	operating	GBR	NaN	London	...	0	0	0	0	0	1	1	Not get funding	yes	no
9	/organization/01games-technology	01Games Technology	http://www.01games.hk/	|Games|	Games	41250.00	operating	HKG	NaN	Hong Kong	...	0	0	0	0	0	1	1	Not get funding	yes	no
10	/organization/1-2-3-listo	1,2,3 Listo	http://www.123listo.com	|E-Commerce|	E-Commerce	40000.00	operating	CHL	NaN	Santiago	...	0	0	0	0	0	1	1	Not get funding	yes	no
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
49428	/organization/zynstra	Zynstra	http://www.zynstra.com	|Software|	Software	14750000.00	operating	GBR	NaN	Bath	...	0	0	0	0	0	1	1	Get funding	yes	yes
49431	/organization/zyrra	Zyrra	http://www.zyrra.com	|E-Commerce|	E-Commerce	1510500.00	operating	USA	MA	Boston	...	0	0	0	0	0	1	1	Get funding	yes	yes
49433	/organization/zzish	Zzish	http://www.zzish.com	|Analytics|Gamification|Developer APIs|iOS|And...	Education	320000.00	operating	GBR	NaN	London	...	0	0	0	0	0	1	1	Not get funding	yes	no
49435	/organization/zzzzapp-com	Zzzzapp Wireless ltd.	http://www.zzzzapp.com	|Web Development|Advertising|Wireless|Mobile|	Web Development	97398.00	operating	HRV	NaN	Split	...	0	0	0	0	0	1	1	Not get funding	yes	no
49436	/organization/a-list-games	[a]list games	http://www.alistgames.com	|Games|	Games	9300000.00	operating	NaN	NaN	NaN	...	0	0	0	0	0	1	1	Not get funding	yes	no
13839 rows × 44 columns

g1=data.groupby('venture_YN')
print(g1.groups)
group_venture=g1.get_group('yes')
group_venture
{'no': Int64Index([    0,     2,     3,     4,     6,     8,     9,    10,    11,
               12,
            ...
            49411, 49413, 49414, 49418, 49421, 49422, 49423, 49433, 49435,
            49436],
           dtype='int64', length=26160), 'yes': Int64Index([    1,     5,     7,    15,    20,    21,    22,    23,    24,
               25,
            ...
            49425, 49426, 49427, 49428, 49429, 49430, 49431, 49432, 49434,
            49437],
           dtype='int64', length=23277)}
permalink	name	homepage_url	category_list	market	funding_total_usd	status	country_code	state_code	region	...	round_D	round_E	round_F	round_G	round_H	count	get_funding_in_seed	get_funding_in_angel	seed_YN	venture_YN
1	/organization/tv-communications	&TV Communications	http://enjoyandtv.com	|Games|	Games	4000000.00	operating	USA	CA	Los Angeles	...	0	0	0	0	0	1	0	Not get funding	no	yes
5	/organization/club-domains	.Club Domains	http://nic.club/	|Software|	Software	7000000.00	NaN	USA	FL	Ft. Lauderdale	...	0	0	0	0	0	1	0	Not get funding	no	yes
7	/organization/0-6-com	0-6.com	http://www.0-6.com	|Curated Web|	Curated Web	2000000.00	operating	NaN	NaN	NaN	...	0	0	0	0	0	1	0	Not get funding	no	yes
15	/organization/10-minutes-with	10 Minutes With	http://10minuteswith.com	|Education|	Education	4400000.00	operating	GBR	NaN	London	...	0	0	0	0	0	1	1	Not get funding	yes	yes
20	/organization/1000memories	1000memories	http://1000memories.com	|Curated Web|	Curated Web	2535000.00	acquired	USA	CA	SF Bay Area	...	0	0	0	0	0	1	1	Not get funding	yes	yes
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
49430	/organization/zyraz-technology	Zyraz Technology	http://www.zyraz.com	|Biotechnology|	Biotechnology	15419877.00	closed	MYS	NaN	MYS - Other	...	0	0	0	0	0	1	0	Get funding	no	yes
49431	/organization/zyrra	Zyrra	http://www.zyrra.com	|E-Commerce|	E-Commerce	1510500.00	operating	USA	MA	Boston	...	0	0	0	0	0	1	1	Get funding	yes	yes
49432	/organization/zytoprotec	Zytoprotec	http://www.zytoprotec.com	|Biotechnology|	Biotechnology	2686600.00	operating	AUT	NaN	Vienna	...	0	0	0	0	0	1	0	Not get funding	no	yes
49434	/organization/zznode-science-and-technology-co...	ZZNode Science and Technology	http://www.zznode.com	|Enterprise Software|	Enterprise Software	1587301.00	operating	CHN	NaN	Beijing	...	0	0	0	0	0	1	0	Not get funding	no	yes
49437	/organization/x	[x+1]	http://www.xplusone.com/	|Enterprise Software|	Enterprise Software	45000000.00	operating	USA	NY	New York City	...	0	0	0	0	0	1	0	Not get funding	no	yes
23277 rows × 44 columns

datas=group_seed[['founded_year', 'seed_YN']]
datas.head()
founded_year	seed_YN
0	2012.00	yes
2	2012.00	yes
3	2011.00	yes
9	nan	yes
10	2012.00	yes
datsa= datas.groupby(['founded_year'])['seed_YN'].count()
datas
founded_year	seed_YN
0	2012.00	yes
2	2012.00	yes
3	2011.00	yes
9	nan	yes
10	2012.00	yes
...	...	...
49428	2011.00	yes
49431	nan	yes
49433	2013.00	yes
49435	2012.00	yes
49436	nan	yes
13839 rows × 2 columns

datav=group_venture[['founded_year', 'venture_YN']]
datav.head()
founded_year	venture_YN
1	nan	yes
5	2011.00	yes
7	2007.00	yes
15	2013.00	yes
20	2010.00	yes
datav = datav.groupby(['founded_year'])['venture_YN'].count()
#datav.sort_values(ascending=False)
datav
founded_year
1902.00       1
1906.00       4
1913.00       1
1919.00       1
1921.00       2
           ... 
2010.00    1646
2011.00    1676
2012.00    1350
2013.00     687
2014.00     116
Name: venture_YN, Length: 77, dtype: int64
#final = data.apply(pd.value_counts).fillna(0)
#final
datas=pd.DataFrame(datas)
datav=pd.DataFrame(datav)
sns.set()
plt.rcParams['figure.figsize'] = 15,6
datav.plot()
<matplotlib.axes._subplots.AxesSubplot at 0x1eb22ad61d0>

plt.rcParams['figure.figsize'] = 15,6
datas.plot()
<matplotlib.axes._subplots.AxesSubplot at 0x1eb22832da0>

dataset=data[["name","seed","venture","angel","grant","seed_YN","venture_YN"]]
dataset.head()
​
name	seed	venture	angel	grant	seed_YN	venture_YN
0	#waywire	1750000	0	0	0	yes	no
1	&TV Communications	0	4000000	0	0	no	yes
2	'Rock' Your Paper	40000	0	0	0	yes	no
3	(In)Touch Network	1500000	0	0	0	yes	no
4	-R- Ranch and Mine	0	0	0	0	no	no
dataset=data[["name","seed","venture","angel","grant","seed_YN","venture_YN"]]
dataset.head()
​
name	seed	venture	angel	grant	seed_YN	venture_YN
0	#waywire	1750000	0	0	0	yes	no
1	&TV Communications	0	4000000	0	0	no	yes
2	'Rock' Your Paper	40000	0	0	0	yes	no
3	(In)Touch Network	1500000	0	0	0	yes	no
4	-R- Ranch and Mine	0	0	0	0	no	no
angel='angel',len(dataset['angel'][dataset['angel'] != 0])
venture="venture",len(dataset['venture'][dataset['venture'] != 0])
seed="seed",len(dataset['seed'][dataset['seed'] != 0])
grant="grant",len(dataset['grant'][dataset['grant'] != 0])
equity_crowdfunding="equity_crowdfunding",len(data['equity_crowdfunding'][data['equity_crowdfunding'] != 0])
print(angel,venture,seed,grant,equity_crowdfunding)
​
datat = {'funding_types':['angel','venture','seed','grant','equity_crowdfunding'], 
                                             'count':[3129, 23277, 13840, 1142, 517],
                                             "closed":[ 264, 1164,  771,   36,    0],
                                             "acquired":[ 188, 2580,  591,   33,    4],
                                             "operating":[ 2627, 19030, 12072,   996,   513]}
       
funding_count = pd.DataFrame(datat) 
  
​
funding_count
​
​
('angel', 3129) ('venture', 23277) ('seed', 13839) ('grant', 1142) ('equity_crowdfunding', 522)
funding_types	count	closed	acquired	operating
0	angel	3129	264	188	2627
1	venture	23277	1164	2580	19030
2	seed	13840	771	591	12072
3	grant	1142	36	33	996
4	equity_crowdfunding	517	0	4	513
plt.bar(funding_count["funding_types"], funding_count['count'])
plt.xlabel("funding_types", fontsize=20)
plt.ylabel("count", fontsize=20)
Text(0, 0.5, 'count')

plt.bar(funding_count["funding_types"], funding_count['closed'])
plt.xlabel("Funding_types", fontsize=20)
plt.ylabel("Closed", fontsize=20)
Text(0, 0.5, 'Closed')

plt.bar(funding_count["funding_types"], funding_count['acquired'])
plt.xlabel("Funding_types", fontsize=20)
plt.ylabel("Acquired", fontsize=20)
Text(0, 0.5, 'Acquired')

plt.bar(funding_count["funding_types"], funding_count['operating'])
plt.xlabel("Funding_types", fontsize=20)
plt.ylabel("Operating", fontsize=20)
Text(0, 0.5, 'Operating')

data["status"].value_counts()
corr_matrix = data.corr()
corr_matrix
funding_total_usd	funding_rounds	founded_year	seed	venture	equity_crowdfunding	undisclosed	convertible_note	debt_financing	angel	...	round_A	round_B	round_C	round_D	round_E	round_F	round_G	round_H	count	get_funding_in_seed
funding_total_usd	1.00	0.11	-0.07	-0.00	0.21	-0.00	0.02	0.01	0.90	0.00	...	0.06	0.10	0.13	0.12	0.11	0.09	0.08	0.07	nan	-0.05
funding_rounds	0.11	1.00	-0.06	0.09	0.40	-0.00	0.03	0.02	0.02	0.06	...	0.17	0.28	0.30	0.20	0.20	0.10	0.06	0.04	nan	0.03
founded_year	-0.07	-0.06	1.00	0.08	-0.09	0.01	-0.04	-0.01	-0.03	0.02	...	-0.02	-0.04	-0.05	-0.03	-0.03	-0.01	-0.00	-0.00	nan	0.27
seed	-0.00	0.09	0.08	1.00	-0.01	-0.00	-0.00	-0.00	-0.00	-0.00	...	0.01	0.00	-0.00	-0.01	-0.01	-0.01	-0.00	-0.00	nan	0.33
venture	0.21	0.40	-0.09	-0.01	1.00	-0.01	0.01	0.00	0.01	0.01	...	0.33	0.50	0.58	0.59	0.53	0.43	0.42	0.37	nan	-0.12
equity_crowdfunding	-0.00	-0.00	0.01	-0.00	-0.01	1.00	-0.00	-0.00	-0.00	0.02	...	-0.00	-0.01	-0.00	-0.00	-0.00	-0.00	-0.00	-0.00	nan	-0.01
undisclosed	0.02	0.03	-0.04	-0.00	0.01	-0.00	1.00	-0.00	-0.00	0.00	...	0.00	-0.00	0.00	0.00	0.03	-0.00	-0.00	-0.00	nan	-0.02
convertible_note	0.01	0.02	-0.01	-0.00	0.00	-0.00	-0.00	1.00	0.00	-0.00	...	-0.00	0.00	0.00	0.00	0.00	-0.00	-0.00	-0.00	nan	-0.01
debt_financing	0.90	0.02	-0.03	-0.00	0.01	-0.00	-0.00	0.00	1.00	-0.00	...	-0.00	0.01	0.01	0.00	0.01	0.01	0.00	-0.00	nan	-0.01
angel	0.00	0.06	0.02	-0.00	0.01	0.02	0.00	-0.00	-0.00	1.00	...	0.02	0.00	0.00	0.01	-0.00	-0.00	-0.00	0.00	nan	-0.02
grant	0.04	0.01	-0.09	-0.01	0.01	-0.00	-0.00	-0.00	-0.00	-0.00	...	0.00	0.00	0.01	0.00	0.01	-0.00	-0.00	-0.00	nan	-0.02
private_equity	0.23	0.06	-0.06	-0.01	0.06	-0.00	0.01	0.01	0.01	0.00	...	-0.00	0.02	0.07	0.06	0.04	0.03	0.01	0.00	nan	-0.04
post_ipo_equity	0.23	0.02	-0.04	-0.00	0.01	-0.00	0.00	0.00	-0.00	-0.00	...	0.00	0.00	0.01	0.01	0.01	0.00	-0.00	-0.00	nan	-0.01
post_ipo_debt	0.26	-0.00	-0.03	-0.00	-0.00	-0.00	-0.00	0.00	-0.00	-0.00	...	-0.00	-0.00	-0.00	-0.00	-0.00	-0.00	-0.00	-0.00	nan	-0.01
secondary_market	0.04	0.01	-0.01	-0.00	0.06	-0.00	-0.00	-0.00	-0.00	0.00	...	0.00	0.02	0.02	0.02	0.00	0.07	0.16	-0.00	nan	-0.01
round_A	0.06	0.17	-0.02	0.01	0.33	-0.00	0.00	-0.00	-0.00	0.02	...	1.00	0.27	0.12	0.04	0.05	0.02	0.00	-0.00	nan	-0.07
round_B	0.10	0.28	-0.04	0.00	0.50	-0.01	-0.00	0.00	0.01	0.00	...	0.27	1.00	0.35	0.12	0.09	0.04	0.01	0.00	nan	-0.08
round_C	0.13	0.30	-0.05	-0.00	0.58	-0.00	0.00	0.00	0.01	0.00	...	0.12	0.35	1.00	0.34	0.14	0.05	0.02	0.01	nan	-0.07
round_D	0.12	0.20	-0.03	-0.01	0.59	-0.00	0.00	0.00	0.00	0.01	...	0.04	0.12	0.34	1.00	0.24	0.12	0.09	0.07	nan	-0.03
round_E	0.11	0.20	-0.03	-0.01	0.53	-0.00	0.03	0.00	0.01	-0.00	...	0.05	0.09	0.14	0.24	1.00	0.37	0.30	0.30	nan	-0.03
round_F	0.09	0.10	-0.01	-0.01	0.43	-0.00	-0.00	-0.00	0.01	-0.00	...	0.02	0.04	0.05	0.12	0.37	1.00	0.24	0.15	nan	-0.02
round_G	0.08	0.06	-0.00	-0.00	0.42	-0.00	-0.00	-0.00	0.00	-0.00	...	0.00	0.01	0.02	0.09	0.30	0.24	1.00	0.86	nan	-0.01
round_H	0.07	0.04	-0.00	-0.00	0.37	-0.00	-0.00	-0.00	-0.00	0.00	...	-0.00	0.00	0.01	0.07	0.30	0.15	0.86	1.00	nan	-0.00
count	nan	nan	nan	nan	nan	nan	nan	nan	nan	nan	...	nan	nan	nan	nan	nan	nan	nan	nan	nan	nan
get_funding_in_seed	-0.05	0.03	0.27	0.33	-0.12	-0.01	-0.02	-0.01	-0.01	-0.02	...	-0.07	-0.08	-0.07	-0.03	-0.03	-0.02	-0.01	-0.00	nan	1.00
25 rows × 25 columns

corr_matrix["venture"].sort_values(ascending=False)
venture                1.00
round_D                0.59
round_C                0.58
round_E                0.53
round_B                0.50
round_F                0.43
round_G                0.42
funding_rounds         0.40
round_H                0.37
round_A                0.33
funding_total_usd      0.21
secondary_market       0.06
private_equity         0.06
post_ipo_equity        0.01
undisclosed            0.01
debt_financing         0.01
grant                  0.01
angel                  0.01
convertible_note       0.00
post_ipo_debt         -0.00
equity_crowdfunding   -0.01
seed                  -0.01
founded_year          -0.09
get_funding_in_seed   -0.12
count                   nan
Name: venture, dtype: float64
ML Models
plt.subplots(figsize=(20,15))
​
sns.heatmap(data.corr(), annot=True, linewidth=0.5);

import sklearn.metrics as metrics
linearRegressor = LinearRegression()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn import utils
import graphviz
from sklearn.ensemble import IsolationForest
df=data
df_status = pd.get_dummies(df["status"],drop_first=True,prefix='s')
df_concat = pd.concat([df, df_status], axis=1)
print (df_concat.head())
                         permalink                name  \
0            /organization/waywire            #waywire   
1  /organization/tv-communications  &TV Communications   
2    /organization/rock-your-paper   'Rock' Your Paper   
3   /organization/in-touch-network   (In)Touch Network   
4   /organization/r-ranch-and-mine  -R- Ranch and Mine   

                    homepage_url  \
0         http://www.waywire.com   
1          http://enjoyandtv.com   
2   http://www.rockyourpaper.org   
3  http://www.InTouchNetwork.com   
4                            NaN   

                                       category_list         market  \
0         |Entertainment|Politics|Social Media|News|          News    
1                                            |Games|         Games    
2                             |Publishing|Education|    Publishing    
3  |Electronics|Guides|Coffee|Restaurants|Music|i...   Electronics    
4                      |Tourism|Entertainment|Games|       Tourism    

   funding_total_usd     status country_code state_code         region  ...  \
0         1750000.00   acquired          USA         NY  New York City  ...   
1         4000000.00  operating          USA         CA    Los Angeles  ...   
2           40000.00  operating          EST        NaN        Tallinn  ...   
3         1500000.00  operating          GBR        NaN         London  ...   
4           60000.00  operating          USA         TX         Dallas  ...   

  round_F  round_G round_H count get_funding_in_seed  get_funding_in_angel  \
0       0        0       0     1                   1       Not get funding   
1       0        0       0     1                   0       Not get funding   
2       0        0       0     1                   1       Not get funding   
3       0        0       0     1                   1       Not get funding   
4       0        0       0     1                   0       Not get funding   

  seed_YN venture_YN  s_closed  s_operating  
0     yes         no         0            0  
1      no        yes         0            1  
2     yes         no         0            1  
3     yes         no         0            1  
4      no         no         0            1  

[5 rows x 46 columns]
df_concat.head()
permalink	name	homepage_url	category_list	market	funding_total_usd	status	country_code	state_code	region	...	round_F	round_G	round_H	count	get_funding_in_seed	get_funding_in_angel	seed_YN	venture_YN	s_closed	s_operating
0	/organization/waywire	#waywire	http://www.waywire.com	|Entertainment|Politics|Social Media|News|	News	1750000.00	acquired	USA	NY	New York City	...	0	0	0	1	1	Not get funding	yes	no	0	0
1	/organization/tv-communications	&TV Communications	http://enjoyandtv.com	|Games|	Games	4000000.00	operating	USA	CA	Los Angeles	...	0	0	0	1	0	Not get funding	no	yes	0	1
2	/organization/rock-your-paper	'Rock' Your Paper	http://www.rockyourpaper.org	|Publishing|Education|	Publishing	40000.00	operating	EST	NaN	Tallinn	...	0	0	0	1	1	Not get funding	yes	no	0	1
3	/organization/in-touch-network	(In)Touch Network	http://www.InTouchNetwork.com	|Electronics|Guides|Coffee|Restaurants|Music|i...	Electronics	1500000.00	operating	GBR	NaN	London	...	0	0	0	1	1	Not get funding	yes	no	0	1
4	/organization/r-ranch-and-mine	-R- Ranch and Mine	NaN	|Tourism|Entertainment|Games|	Tourism	60000.00	operating	USA	TX	Dallas	...	0	0	0	1	0	Not get funding	no	no	0	1
5 rows × 46 columns

df_concat[["s_closed","s_operating"]].isnull
#df_concat.fillna(method='bfill')
<bound method DataFrame.isnull of        s_closed  s_operating
0             0            0
1             0            1
2             0            1
3             0            1
4             0            1
...         ...          ...
49433         0            1
49434         0            1
49435         0            1
49436         0            1
49437         0            1

[49437 rows x 2 columns]>
here we have converted Status to a catagorial values

Now we are using Decision tree to see is there any relation between funding and operating status

xt=df_concat[["s_closed","s_operating"]]
xt=df_concat[["s_closed","s_operating"]]#.values.reshape(-1,1)
​
#xt.fillna(xt.mean())
funding_total_usd
yt=data["funding_total_usd"]#.values.reshape(-1,1)
#yt.fillna(yt.mean())
X_train, X_test, y_train, y_test = train_test_split(xt, yt, test_size = 0.3)
clf = tree.DecisionTreeClassifier(criterion="entropy")
X_train, X_test, y_train, y_test = train_test_split(xt, yt, test_size = 0.3)
clf = tree.DecisionTreeClassifier(criterion="entropy")
X_test.fillna(X_test.mean(),inplace=True)
y_test.fillna(y_test.mean(),inplace=True)
X_train.fillna(X_train.mean(),inplace=True)
y_train.fillna(y_train.mean(),inplace=True)
lab_enc = preprocessing.LabelEncoder()
ytrain = lab_enc.fit_transform(y_train)
print(ytrain)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(ytrain))
​
[7391 7964 7964 ... 7964 7890 7964]
continuous
multiclass
multiclass
clf = tree.DecisionTreeClassifier(criterion="entropy")
​
clf.fit(X_train,ytrain)
clf.fit(X_train,ytrain)
DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
tree.plot_tree(clf)
tree.plot_tree(clf)
[Text(502.20000000000005, 271.8, 'X[1] <= 0.5\nentropy = 9.668\nsamples = 34605\nvalue = [1, 1, 1 ... 1, 1, 1]'),
 Text(334.8, 163.08, 'X[0] <= 0.5\nentropy = 8.772\nsamples = 5260\nvalue = [0, 0, 0 ... 1, 0, 0]'),
 Text(167.4, 54.360000000000014, 'entropy = 8.811\nsamples = 3472\nvalue = [0, 0, 0 ... 1, 0, 0]'),
 Text(502.20000000000005, 54.360000000000014, 'entropy = 7.566\nsamples = 1788\nvalue = [0, 0, 0 ... 0, 0, 0]'),
 Text(669.6, 163.08, 'entropy = 9.596\nsamples = 29345\nvalue = [1, 1, 1 ... 0, 1, 1]')]

dot_data=tree.export_graphviz(clf,out_file=None)
graph=graphviz.Source(dot_data)
dot_data = tree.export_graphviz(clf, out_file=None,filled=True, rounded=True, special_characters=True)  
graph = graphviz.Source(dot_data) 
graph
dot_data=tree.export_graphviz(clf,out_file=None)
graph=graphviz.Source(dot_data)
dot_data = tree.export_graphviz(clf, out_file=None,filled=True, rounded=True, special_characters=True)  
graph = graphviz.Source(dot_data) 
graph
X
1
≤ 0.5
entropy = 9.668
samples = 34605
value = [1, 1, 1 ... 1, 1, 1]
X
0
≤ 0.5
entropy = 8.772
samples = 5260
value = [0, 0, 0 ... 1, 0, 0]
True
entropy = 9.596
samples = 29345
value = [1, 1, 1 ... 0, 1, 1]
False
entropy = 8.811
samples = 3472
value = [0, 0, 0 ... 1, 0, 0]
entropy = 7.566
samples = 1788
value = [0, 0, 0 ... 0, 0, 0]
clf.score(X_test, y_test)
y_pred=clf.predict(X_test)
#clf.score(X_test, y_test)
AS we can see tree does classify on the basis of total funding and operating

xt=df_concat[["s_closed","s_operating"]]
yt=data[["funding_rounds","funding_total_usd"]]
X_train, X_test, y_train, y_test = train_test_split(xt, yt, test_size = 0.3)
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train,ytrain)
tree.plot_tree(clf)
[Text(502.20000000000005, 271.8, 'X[1] <= 0.5\nentropy = 9.668\nsamples = 34605\nvalue = [1, 1, 1 ... 1, 1, 1]'),
 Text(334.8, 163.08, 'X[0] <= 0.5\nentropy = 8.76\nsamples = 5241\nvalue = [0, 0, 0 ... 0, 0, 0]'),
 Text(167.4, 54.360000000000014, 'entropy = 8.575\nsamples = 3472\nvalue = [0, 0, 0 ... 0, 0, 0]'),
 Text(502.20000000000005, 54.360000000000014, 'entropy = 8.043\nsamples = 1769\nvalue = [0, 0, 0 ... 0, 0, 0]'),
 Text(669.6, 163.08, 'entropy = 9.599\nsamples = 29364\nvalue = [1, 1, 1 ... 1, 1, 1]')]

dot_data=tree.export_graphviz(clf,out_file=None)
graph=graphviz.Source(dot_data)
dot_data = tree.export_graphviz(clf, out_file=None,filled=True, rounded=True, special_characters=True)  
graph = graphviz.Source(dot_data) 
graph
X
1
≤ 0.5
entropy = 9.668
samples = 34605
value = [1, 1, 1 ... 1, 1, 1]
X
0
≤ 0.5
entropy = 8.76
samples = 5241
value = [0, 0, 0 ... 0, 0, 0]
True
entropy = 9.599
samples = 29364
value = [1, 1, 1 ... 1, 1, 1]
False
entropy = 8.575
samples = 3472
value = [0, 0, 0 ... 0, 0, 0]
entropy = 8.043
samples = 1769
value = [0, 0, 0 ... 0, 0, 0]
#
#clf.score(X_test, y_test)
now we have created tree based on funding round and operating status and total amount of funding

​
​
ytt=data["funding_rounds"]
X_train, X_test, y_train, y_test = train_test_split(xt, ytt, test_size = 0.3)
clf.fit(X_train,ytrain)
DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
tree.plot_tree(clf)
[Text(502.20000000000005, 271.8, 'X[1] <= 0.5\nentropy = 9.668\nsamples = 34605\nvalue = [1, 1, 1 ... 1, 1, 1]'),
 Text(334.8, 163.08, 'X[0] <= 0.5\nentropy = 8.748\nsamples = 5290\nvalue = [1, 1, 1 ... 0, 0, 0]'),
 Text(167.4, 54.360000000000014, 'entropy = 8.462\nsamples = 3475\nvalue = [0, 1, 0 ... 0, 0, 0]'),
 Text(502.20000000000005, 54.360000000000014, 'entropy = 8.205\nsamples = 1815\nvalue = [1, 0, 1 ... 0, 0, 0]'),
 Text(669.6, 163.08, 'entropy = 9.6\nsamples = 29315\nvalue = [0, 0, 0 ... 1, 1, 1]')]

y_pred=clf.predict(X_test)
dot_data=tree.export_graphviz(clf,out_file=None)
graph=graphviz.Source(dot_data)
dot_data = tree.export_graphviz(clf, out_file=None,filled=True, rounded=True, special_characters=True)  
graph = graphviz.Source(dot_data) 
graph
X
1
≤ 0.5
entropy = 9.668
samples = 34605
value = [1, 1, 1 ... 1, 1, 1]
X
0
≤ 0.5
entropy = 8.748
samples = 5290
value = [1, 1, 1 ... 0, 0, 0]
True
entropy = 9.6
samples = 29315
value = [0, 0, 0 ... 1, 1, 1]
False
entropy = 8.462
samples = 3475
value = [0, 1, 0 ... 0, 0, 0]
entropy = 8.205
samples = 1815
value = [1, 0, 1 ... 0, 0, 0]
The above fig shows tree diagram for funding round and operating status

y=data["round_C"].values.reshape(-1,1)
x=data["venture"].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))
​
y=data["round_C"].values.reshape(-1,1)
x=data["venture"].values.reshape(-1,1)
​
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))
              precision    recall  f1-score   support

           0       0.94      1.00      0.97     15362
       76265       0.00      0.00      0.00         1
      225417       0.00      0.00      0.00         1
      272745       0.00      0.00      0.00         1
      303757       0.00      0.00      0.00         1
      440000       0.00      0.00      0.00         1
      450000       0.00      0.00      0.00         1
      500000       0.00      0.00      0.00         3
      506919       0.00      0.00      0.00         1
      532850       0.00      0.00      0.00         1
      687500       0.00      0.00      0.00         1
      785000       0.00      0.00      0.00         1
      800000       0.00      0.00      0.00         1
      825000       0.00      0.00      0.00         1
      925000       0.00      0.00      0.00         1
     1000000       0.00      0.00      0.00         6
     1148040       0.00      0.00      0.00         1
     1200000       0.00      0.00      0.00         1
     1250000       0.00      0.00      0.00         1
     1424336       0.00      0.00      0.00         1
     1450000       0.00      0.00      0.00         1
     1500000       0.00      0.00      0.00         2
     1600000       0.00      0.00      0.00         1
     1776763       0.00      0.00      0.00         1
     1800000       0.00      0.00      0.00         1
     1850000       0.00      0.00      0.00         1
     1870000       0.00      0.00      0.00         1
     1880000       0.00      0.00      0.00         1
     2000000       0.00      0.00      0.00        15
     2048334       0.00      0.00      0.00         1
     2058560       0.00      0.00      0.00         1
     2100000       0.00      0.00      0.00         1
     2307806       0.00      0.00      0.00         1
     2400000       0.00      0.00      0.00         1
     2500000       0.00      0.00      0.00         2
     2508870       0.00      0.00      0.00         1
     2575000       0.00      0.00      0.00         1
     2650000       0.00      0.00      0.00         1
     2660000       0.00      0.00      0.00         1
     2750000       0.00      0.00      0.00         1
     2915715       0.00      0.00      0.00         1
     2975000       0.00      0.00      0.00         1
     3000000       0.00      0.00      0.00        10
     3200000       0.00      0.00      0.00         2
     3270000       0.00      0.00      0.00         1
     3300000       0.00      0.00      0.00         1
     3500000       0.00      0.00      0.00         6
     3600020       0.00      0.00      0.00         1
     3750000       0.00      0.00      0.00         1
     3880000       0.00      0.00      0.00         1
     3912510       0.00      0.00      0.00         1
     3988460       0.00      0.00      0.00         1
     4000000       0.00      0.00      0.00        16
     4100000       0.00      0.00      0.00         1
     4125014       0.00      0.00      0.00         1
     4250000       0.00      0.00      0.00         3
     4340000       0.00      0.00      0.00         1
     4490000       0.00      0.00      0.00         1
     4580000       0.00      0.00      0.00         1
     4722711       0.00      0.00      0.00         1
     4751640       0.00      0.00      0.00         1
     4760000       0.00      0.00      0.00         1
     4853100       0.00      0.00      0.00         1
     4900000       0.00      0.00      0.00         1
     4950000       0.00      0.00      0.00         1
     5000000       0.00      0.00      0.00        31
     5070000       0.00      0.00      0.00         1
     5165160       0.00      0.00      0.00         1
     5200000       0.00      0.00      0.00         1
     5268902       0.00      0.00      0.00         1
     5270000       0.00      0.00      0.00         1
     5300000       0.00      0.00      0.00         1
     5378509       0.00      0.00      0.00         1
     5400000       0.00      0.00      0.00         1
     5500000       0.00      0.00      0.00         7
     5600000       0.00      0.00      0.00         2
     5604719       0.00      0.00      0.00         1
     5625000       0.00      0.00      0.00         1
     5650000       0.00      0.00      0.00         1
     5750000       0.00      0.00      0.00         1
     5760000       0.00      0.00      0.00         1
     5800000       0.00      0.00      0.00         1
     5848920       0.00      0.00      0.00         1
     5999999       0.00      0.00      0.00         1
     6000000       0.00      0.00      0.00        19
     6058226       0.00      0.00      0.00         1
     6100000       0.00      0.00      0.00         1
     6200000       0.00      0.00      0.00         1
     6240000       0.00      0.00      0.00         1
     6300000       0.00      0.00      0.00         3
     6321218       0.00      0.00      0.00         1
     6351930       0.00      0.00      0.00         1
     6360000       0.00      0.00      0.00         1
     6400000       0.00      0.00      0.00         1
     6403060       0.00      0.00      0.00         1
     6440000       0.00      0.00      0.00         1
     6500000       0.00      0.00      0.00         4
     6564658       0.00      0.00      0.00         1
     6600000       0.00      0.00      0.00         1
     6600306       0.00      0.00      0.00         1
     6700000       0.00      0.00      0.00         1
     6820000       0.00      0.00      0.00         1
     6999684       0.00      0.00      0.00         1
     7000000       0.00      0.00      0.00        13
     7200000       0.00      0.00      0.00         1
     7393665       0.00      0.00      0.00         1
     7400000       0.00      0.00      0.00         2
     7500000       0.00      0.00      0.00         6
     7557394       0.00      0.00      0.00         1
     7576257       0.00      0.00      0.00         1
     7600000       0.00      0.00      0.00         1
     7760573       0.00      0.00      0.00         1
     7800000       0.00      0.00      0.00         2
     7910000       0.00      0.00      0.00         1
     8000000       0.00      0.00      0.00        20
     8100000       0.00      0.00      0.00         1
     8225000       0.00      0.00      0.00         1
     8300000       0.00      0.00      0.00         2
     8400000       0.00      0.00      0.00         1
     8500000       0.00      0.00      0.00         4
     8520000       0.00      0.00      0.00         1
     8600000       0.00      0.00      0.00         1
     8700000       0.00      0.00      0.00         1
     8750000       0.00      0.00      0.00         1
     8798583       0.00      0.00      0.00         1
     8800000       0.00      0.00      0.00         1
     8832072       0.00      0.00      0.00         1
     8864965       0.00      0.00      0.00         1
     9000000       0.00      0.00      0.00        15
     9020000       0.00      0.00      0.00         1
     9250000       0.00      0.00      0.00         1
     9300000       0.00      0.00      0.00         1
     9353400       0.00      0.00      0.00         1
     9429020       0.00      0.00      0.00         1
     9500000       0.00      0.00      0.00         3
     9600000       0.00      0.00      0.00         1
     9782500       0.00      0.00      0.00         1
     9996147       0.00      0.00      0.00         1
    10000000       0.00      0.00      0.00        48
    10000001       0.00      0.00      0.00         1
    10000006       0.00      0.00      0.00         1
    10021165       0.00      0.00      0.00         1
    10100000       0.00      0.00      0.00         1
    10200000       0.00      0.00      0.00         1
    10300000       0.00      0.00      0.00         3
    10400000       0.00      0.00      0.00         1
    10500000       0.00      0.00      0.00         2
    10600000       0.00      0.00      0.00         1
    10700000       0.00      0.00      0.00         2
    10850000       0.00      0.00      0.00         1
    10870000       0.00      0.00      0.00         1
    10973011       0.00      0.00      0.00         1
    11000000       0.00      0.00      0.00         8
    11250000       0.00      0.00      0.00         1
    11500000       0.00      0.00      0.00         3
    11600000       0.00      0.00      0.00         2
    11750000       0.00      0.00      0.00         2
    11800000       0.00      0.00      0.00         1
    11900000       0.00      0.00      0.00         1
    12000000       0.00      0.00      0.00        27
    12100000       0.00      0.00      0.00         1
    12122011       0.00      0.00      0.00         1
    12200000       0.00      0.00      0.00         1
    12300000       0.00      0.00      0.00         2
    12500000       0.00      0.00      0.00         7
    12600000       0.00      0.00      0.00         3
    12639999       0.00      0.00      0.00         1
    12700000       0.00      0.00      0.00         1
    13000000       0.00      0.00      0.00        14
    13080000       0.00      0.00      0.00         1
    13100000       0.00      0.00      0.00         1
    13200000       0.00      0.00      0.00         1
    13500000       0.00      0.00      0.00         3
    13600000       0.00      0.00      0.00         1
    13612500       0.00      0.00      0.00         1
    13700000       0.00      0.00      0.00         1
    13900000       0.00      0.00      0.00         1
    14000000       0.00      0.00      0.00         6
    14200000       0.00      0.00      0.00         1
    14380000       0.00      0.00      0.00         1
    14500000       0.00      0.00      0.00         1
    14571510       0.00      0.00      0.00         1
    14600000       0.00      0.00      0.00         1
    14700000       0.00      0.00      0.00         1
    14800000       0.00      0.00      0.00         1
    14849130       0.00      0.00      0.00         1
    15000000       0.00      0.00      0.00        52
    15100000       0.00      0.00      0.00         1
    15200000       0.00      0.00      0.00         4
    15500000       0.00      0.00      0.00         4
    15600000       0.00      0.00      0.00         1
    15985452       0.00      0.00      0.00         1
    16000000       0.00      0.00      0.00         7
    16164879       0.00      0.00      0.00         1
    16236400       0.00      0.00      0.00         3
    16450000       0.00      0.00      0.00         1
    16500000       0.00      0.00      0.00         6
    16700000       0.00      0.00      0.00         1
    16800000       0.00      0.00      0.00         1
    16939031       0.00      0.00      0.00         1
    17000000       0.00      0.00      0.00         9
    17226000       0.00      0.00      0.00         1
    17270000       0.00      0.00      0.00         1
    17500000       0.00      0.00      0.00         4
    17600000       0.00      0.00      0.00         1
    17610000       0.00      0.00      0.00         1
    17892602       0.00      0.00      0.00         1
    18000000       0.00      0.00      0.00        13
    18012400       0.00      0.00      0.00         1
    18200000       0.00      0.00      0.00         2
    18250000       0.00      0.00      0.00         1
    18400000       0.00      0.00      0.00         1
    18500000       0.00      0.00      0.00         1
    18770000       0.00      0.00      0.00         1
    18900000       0.00      0.00      0.00         1
    19000000       0.00      0.00      0.00         6
    19300000       0.00      0.00      0.00         1
    19480000       0.00      0.00      0.00         1
    19491529       0.00      0.00      0.00         1
    19500000       0.00      0.00      0.00         2
    20000000       0.00      0.00      0.00        44
    20100000       0.00      0.00      0.00         2
    20553000       0.00      0.00      0.00         1
    21000000       0.00      0.00      0.00         7
    21193815       0.00      0.00      0.00         1
    21300000       0.00      0.00      0.00         1
    21500000       0.00      0.00      0.00         1
    21800000       0.00      0.00      0.00         1
    21858476       0.00      0.00      0.00         1
    22000000       0.00      0.00      0.00        11
    22100000       0.00      0.00      0.00         1
    22200000       0.00      0.00      0.00         1
    22500000       0.00      0.00      0.00         2
    23000000       0.00      0.00      0.00         4
    23196677       0.00      0.00      0.00         1
    23452382       0.00      0.00      0.00         1
    24000000       0.00      0.00      0.00         6
    24400000       0.00      0.00      0.00         1
    24500000       0.00      0.00      0.00         4
    24801993       0.00      0.00      0.00         1
    25000000       0.00      0.00      0.00        31
    25000016       0.00      0.00      0.00         1
    25090000       0.00      0.00      0.00         1
    25200000       0.00      0.00      0.00         1
    25280000       0.00      0.00      0.00         1
    25500000       0.00      0.00      0.00         3
    26000000       0.00      0.00      0.00         5
    26772146       0.00      0.00      0.00         1
    27000000       0.00      0.00      0.00         3
    27013076       0.00      0.00      0.00         1
    27200000       0.00      0.00      0.00         1
    27300000       0.00      0.00      0.00         1
    27400000       0.00      0.00      0.00         1
    27700000       0.00      0.00      0.00         1
    28000000       0.00      0.00      0.00         3
    28200000       0.00      0.00      0.00         1
    28314585       0.00      0.00      0.00         1
    28700000       0.00      0.00      0.00         1
    30000000       0.00      0.00      0.00        28
    30300000       0.00      0.00      0.00         1
    30500000       0.00      0.00      0.00         1
    31444456       0.00      0.00      0.00         1
    31600000       0.00      0.00      0.00         1
    32000000       0.00      0.00      0.00         3
    32472800       0.00      0.00      0.00         1
    32500000       0.00      0.00      0.00         1
    33000000       0.00      0.00      0.00         4
    33500000       0.00      0.00      0.00         1
    34000000       0.00      0.00      0.00         1
    34999994       0.00      0.00      0.00         1
    35000000       0.00      0.00      0.00        17
    35500000       0.00      0.00      0.00         1
    36000000       0.00      0.00      0.00         3
    36500000       0.00      0.00      0.00         1
    36600000       0.00      0.00      0.00         1
    37000000       0.00      0.00      0.00         1
    38000000       0.00      0.00      0.00         3
    38097364       0.00      0.00      0.00         1
    39000000       0.00      0.00      0.00         1
    40000000       0.00      0.00      0.00        17
    41000000       0.00      0.00      0.00         2
    41500000       0.00      0.00      0.00         2
    42000000       0.00      0.00      0.00         4
    43000000       0.00      0.00      0.00         2
    44000000       0.00      0.00      0.00         1
    44500000       0.00      0.00      0.00         1
    44600000       0.00      0.00      0.00         1
    45000000       0.00      0.00      0.00         2
    45353700       0.00      0.00      0.00         1
    46100000       0.00      0.00      0.00         1
    46400000       0.00      0.00      0.00         1
    46500000       0.00      0.00      0.00         1
    49250000       0.00      0.00      0.00         1
    50000000       0.00      0.00      0.00        13
    50600000       0.00      0.00      0.00         1
    51000000       0.00      0.00      0.00         1
    51498432       0.00      0.00      0.00         1
    52000000       0.00      0.00      0.00         1
    52300000       0.00      0.00      0.00         1
    52700000       0.00      0.00      0.00         1
    53000000       0.00      0.00      0.00         2
    54000000       0.00      0.00      0.00         3
    55000000       0.00      0.00      0.00         5
    56000000       0.00      0.00      0.00         2
    57000000       0.00      0.00      0.00         1
    58000000       0.00      0.00      0.00         2
    58899155       0.00      0.00      0.00         1
    60000000       0.00      0.00      0.00         2
    62000000       0.00      0.00      0.00         1
    65000000       0.00      0.00      0.00         5
    65740000       0.00      0.00      0.00         1
    66000000       0.00      0.00      0.00         1
    68000000       0.00      0.00      0.00         1
    70000000       0.00      0.00      0.00         6
    70848000       0.00      0.00      0.00         1
    72000000       0.00      0.00      0.00         1
    73500000       0.00      0.00      0.00         1
    75000000       0.00      0.00      0.00         1
    77700000       0.00      0.00      0.00         1
    80000000       0.00      0.00      0.00         2
    85000000       0.00      0.00      0.00         1
    89000000       0.00      0.00      0.00         1
    94357588       0.00      0.00      0.00         1
    95000000       0.00      0.00      0.00         1
    95708000       0.00      0.00      0.00         1
    97000000       0.00      0.00      0.00         1
    98000000       0.00      0.00      0.00         1
   100000000       0.00      0.00      0.00         6
   103500000       0.00      0.00      0.00         1
   104000000       0.00      0.00      0.00         1
   105000000       0.00      0.00      0.00         1
   115000000       0.00      0.00      0.00         1
   120000000       0.00      0.00      0.00         1
   125000000       0.00      0.00      0.00         1
   130000000       0.00      0.00      0.00         1
   135000000       0.00      0.00      0.00         1
   150000000       0.00      0.00      0.00         1
   155000000       0.00      0.00      0.00         1
   200000000       0.00      0.00      0.00         1
   375000000       0.00      0.00      0.00         1

    accuracy                           0.94     16315
   macro avg       0.00      0.00      0.00     16315
weighted avg       0.89      0.94      0.91     16315

[[15362     0     0 ...     0     0     0]
 [    1     0     0 ...     0     0     0]
 [    1     0     0 ...     0     0     0]
 ...
 [    1     0     0 ...     0     0     0]
 [    1     0     0 ...     0     0     0]
 [    1     0     0 ...     0     0     0]]
0.9415874961691695
veture funding and round c funding are similar but they are not the same , so we have used logisctics regression to classify

labelencoder = LabelEncoder()
data["market"]=data['market'].astype('category')
#data["market"]=data['market'].astype('category')
data['market_Cat'] = data['market'].cat.codes
data
permalink	name	homepage_url	category_list	market	funding_total_usd	status	country_code	state_code	region	...	round_E	round_F	round_G	round_H	count	get_funding_in_seed	get_funding_in_angel	seed_YN	venture_YN	market_Cat
0	/organization/waywire	#waywire	http://www.waywire.com	|Entertainment|Politics|Social Media|News|	News	1750000.00	acquired	USA	NY	New York City	...	0	0	0	0	1	1	Not get funding	yes	no	465
1	/organization/tv-communications	&TV Communications	http://enjoyandtv.com	|Games|	Games	4000000.00	operating	USA	CA	Los Angeles	...	0	0	0	0	1	0	Not get funding	no	yes	277
2	/organization/rock-your-paper	'Rock' Your Paper	http://www.rockyourpaper.org	|Publishing|Education|	Publishing	40000.00	operating	EST	NaN	Tallinn	...	0	0	0	0	1	1	Not get funding	yes	no	543
3	/organization/in-touch-network	(In)Touch Network	http://www.InTouchNetwork.com	|Electronics|Guides|Coffee|Restaurants|Music|i...	Electronics	1500000.00	operating	GBR	NaN	London	...	0	0	0	0	1	1	Not get funding	yes	no	211
4	/organization/r-ranch-and-mine	-R- Ranch and Mine	NaN	|Tourism|Entertainment|Games|	Tourism	60000.00	operating	USA	TX	Dallas	...	0	0	0	0	1	0	Not get funding	no	no	683
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
49433	/organization/zzish	Zzish	http://www.zzish.com	|Analytics|Gamification|Developer APIs|iOS|And...	Education	320000.00	operating	GBR	NaN	London	...	0	0	0	0	1	1	Not get funding	yes	no	205
49434	/organization/zznode-science-and-technology-co...	ZZNode Science and Technology	http://www.zznode.com	|Enterprise Software|	Enterprise Software	1587301.00	operating	CHN	NaN	Beijing	...	0	0	0	0	1	0	Not get funding	no	yes	233
49435	/organization/zzzzapp-com	Zzzzapp Wireless ltd.	http://www.zzzzapp.com	|Web Development|Advertising|Wireless|Mobile|	Web Development	97398.00	operating	HRV	NaN	Split	...	0	0	0	0	1	1	Not get funding	yes	no	732
49436	/organization/a-list-games	[a]list games	http://www.alistgames.com	|Games|	Games	9300000.00	operating	NaN	NaN	NaN	...	0	0	0	0	1	1	Not get funding	yes	no	277
49437	/organization/x	[x+1]	http://www.xplusone.com/	|Enterprise Software|	Enterprise Software	45000000.00	operating	USA	NY	New York City	...	0	0	0	0	1	0	Not get funding	no	yes	233
49437 rows × 45 columns

data["country_code"]=data['country_code'].astype('category')
data['ccountry_code_cat'] = data['country_code'].cat.codes
data
permalink	name	homepage_url	category_list	market	funding_total_usd	status	country_code	state_code	region	...	round_F	round_G	round_H	count	get_funding_in_seed	get_funding_in_angel	seed_YN	venture_YN	market_Cat	ccountry_code_cat
0	/organization/waywire	#waywire	http://www.waywire.com	|Entertainment|Politics|Social Media|News|	News	1750000.00	acquired	USA	NY	New York City	...	0	0	0	1	1	Not get funding	yes	no	465	110
1	/organization/tv-communications	&TV Communications	http://enjoyandtv.com	|Games|	Games	4000000.00	operating	USA	CA	Los Angeles	...	0	0	0	1	0	Not get funding	no	yes	277	110
2	/organization/rock-your-paper	'Rock' Your Paper	http://www.rockyourpaper.org	|Publishing|Education|	Publishing	40000.00	operating	EST	NaN	Tallinn	...	0	0	0	1	1	Not get funding	yes	no	543	35
3	/organization/in-touch-network	(In)Touch Network	http://www.InTouchNetwork.com	|Electronics|Guides|Coffee|Restaurants|Music|i...	Electronics	1500000.00	operating	GBR	NaN	London	...	0	0	0	1	1	Not get funding	yes	no	211	38
4	/organization/r-ranch-and-mine	-R- Ranch and Mine	NaN	|Tourism|Entertainment|Games|	Tourism	60000.00	operating	USA	TX	Dallas	...	0	0	0	1	0	Not get funding	no	no	683	110
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
49433	/organization/zzish	Zzish	http://www.zzish.com	|Analytics|Gamification|Developer APIs|iOS|And...	Education	320000.00	operating	GBR	NaN	London	...	0	0	0	1	1	Not get funding	yes	no	205	38
49434	/organization/zznode-science-and-technology-co...	ZZNode Science and Technology	http://www.zznode.com	|Enterprise Software|	Enterprise Software	1587301.00	operating	CHN	NaN	Beijing	...	0	0	0	1	0	Not get funding	no	yes	233	20
49435	/organization/zzzzapp-com	Zzzzapp Wireless ltd.	http://www.zzzzapp.com	|Web Development|Advertising|Wireless|Mobile|	Web Development	97398.00	operating	HRV	NaN	Split	...	0	0	0	1	1	Not get funding	yes	no	732	44
49436	/organization/a-list-games	[a]list games	http://www.alistgames.com	|Games|	Games	9300000.00	operating	NaN	NaN	NaN	...	0	0	0	1	1	Not get funding	yes	no	277	-1
49437	/organization/x	[x+1]	http://www.xplusone.com/	|Enterprise Software|	Enterprise Software	45000000.00	operating	USA	NY	New York City	...	0	0	0	1	0	Not get funding	no	yes	233	110
49437 rows × 46 columns

converting country code and market into catagorical values to form clusters

from sklearn.cluster import DBSCAN
M=data["market_Cat"].values.reshape(-1,1)
neigh = NearestNeighbors(n_neighbors=4)
nbrs = neigh.fit(M)
distances, indices = nbrs.kneighbors(M)
clustering = DBSCAN(eps=1.5, min_samples=5).fit(M)
print(clustering.labels_)
labels = clustering.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
[0 1 0 ... 3 1 2]
Estimated number of clusters: 8
Estimated number of noise points: 0
we have made cluster using density based clustering and got greate accuracy, we haved used knn to get eps value
we have made cluster using density based clustering and got greate accuracy, we haved used knn to get eps value
M=data["market_Cat"].values.reshape(-1,1)
C=data["ccountry_code_cat"].values.reshape(-1,1)
​
​
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,  random_state=0)
    kmeans.fit(C)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

​
​
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=3)
nbrs = neigh.fit(C)
distances, indices = nbrs.kneighbors(C)
from matplotlib import pyplot as plt
distances = np.sort(distances, axis=0)
#distances = distances[:,1]
plt.plot(distances)
[<matplotlib.lines.Line2D at 0x1eb2231a278>,
 <matplotlib.lines.Line2D at 0x1eb2231a1d0>,
 <matplotlib.lines.Line2D at 0x1eb223c6390>]

from sklearn.cluster import DBSCAN
​
neigh = NearestNeighbors(n_neighbors=4)
nbrs = neigh.fit(C)
distances, indices = nbrs.kneighbors(C)
clustering = DBSCAN(eps=0.4, min_samples=5).fit(C)
print(clustering.labels_)
labels = clustering.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
[ 0  0  1 ... 49  4  0]
Estimated number of clusters: 77
Estimated number of noise points: 76
we have made clusterof city using density based clustering and got greate accuracy, we haved used knn to get eps value

data["funding_total_usd"].fillna(data["funding_total_usd"].mean(),inplace=True)
data["market_Cat"].fillna(data["market_Cat"].mean(),inplace=True)
['
R=data[['seed','venture','angel']].values.reshape(-1,1)
​
​
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,  random_state=0)
    kmeans.fit(R)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=4)
nbrs = neigh.fit(R)
distances, indices = nbrs.kneighbors(R)
from matplotlib import pyplot as plt
distances = np.sort(distances, axis=0)
#distances = distances[:,1]
plt.plot(distances)
[<matplotlib.lines.Line2D at 0x1eb24633d68>,
 <matplotlib.lines.Line2D at 0x1eb24633ef0>,
 <matplotlib.lines.Line2D at 0x1eb246270b8>,
 <matplotlib.lines.Line2D at 0x1eb24627240>]

​
---------------------------------------------------------------------------
MemoryError                               Traceback (most recent call last)
<ipython-input-140-6c9901359d9a> in <module>
----> 1 clustering = DBSCAN(eps=0.4, min_samples=5).fit(R)

~\Anaconda3\lib\site-packages\sklearn\cluster\dbscan_.py in fit(self, X, y, sample_weight)
    349         X = check_array(X, accept_sparse='csr')
    350         clust = dbscan(X, sample_weight=sample_weight,
--> 351                        **self.get_params())
    352         self.core_sample_indices_, self.labels_ = clust
    353         if len(self.core_sample_indices_):

~\Anaconda3\lib\site-packages\sklearn\cluster\dbscan_.py in dbscan(X, eps, min_samples, metric, metric_params, algorithm, leaf_size, p, sample_weight, n_jobs)
    173         # This has worst case O(n^2) memory complexity
    174         neighborhoods = neighbors_model.radius_neighbors(X, eps,
--> 175                                                          return_distance=False)
    176 
    177     if sample_weight is None:

~\Anaconda3\lib\site-packages\sklearn\neighbors\base.py in radius_neighbors(self, X, radius, return_distance)
    745             results = Parallel(n_jobs, **parallel_kwargs)(
    746                 delayed_query(self._tree, X[s], radius, return_distance)
--> 747                 for s in gen_even_slices(X.shape[0], n_jobs)
    748             )
    749             if return_distance:

~\Anaconda3\lib\site-packages\joblib\parallel.py in __call__(self, iterable)
   1001             # remaining jobs.
   1002             self._iterating = False
-> 1003             if self.dispatch_one_batch(iterator):
   1004                 self._iterating = self._original_iterator is not None
   1005 

~\Anaconda3\lib\site-packages\joblib\parallel.py in dispatch_one_batch(self, iterator)
    832                 return False
    833             else:
--> 834                 self._dispatch(tasks)
    835                 return True
    836 

~\Anaconda3\lib\site-packages\joblib\parallel.py in _dispatch(self, batch)
    751         with self._lock:
    752             job_idx = len(self._jobs)
--> 753             job = self._backend.apply_async(batch, callback=cb)
    754             # A job can complete so quickly than its callback is
    755             # called before we get here, causing self._jobs to

~\Anaconda3\lib\site-packages\joblib\_parallel_backends.py in apply_async(self, func, callback)
    199     def apply_async(self, func, callback=None):
    200         """Schedule a func to be run"""
--> 201         result = ImmediateResult(func)
    202         if callback:
    203             callback(result)

~\Anaconda3\lib\site-packages\joblib\_parallel_backends.py in __init__(self, batch)
    580         # Don't delay the application, to avoid keeping the input
    581         # arguments in memory
--> 582         self.results = batch()
    583 
    584     def get(self):

~\Anaconda3\lib\site-packages\joblib\parallel.py in __call__(self)
    254         with parallel_backend(self._backend, n_jobs=self._n_jobs):
    255             return [func(*args, **kwargs)
--> 256                     for func, args, kwargs in self.items]
    257 
    258     def __len__(self):

~\Anaconda3\lib\site-packages\joblib\parallel.py in <listcomp>(.0)
    254         with parallel_backend(self._backend, n_jobs=self._n_jobs):
    255             return [func(*args, **kwargs)
--> 256                     for func, args, kwargs in self.items]
    257 
    258     def __len__(self):

~\Anaconda3\lib\site-packages\sklearn\neighbors\base.py in _tree_query_radius_parallel_helper(tree, data, radius, return_distance)
    578     cloudpickle under PyPy.
    579     """
--> 580     return tree.query_radius(data, radius, return_distance)
    581 
    582 

sklearn\neighbors\binary_tree.pxi in sklearn.neighbors.kd_tree.BinaryTree.query_radius()

sklearn\neighbors\binary_tree.pxi in sklearn.neighbors.kd_tree.BinaryTree.query_radius()

MemoryError: 

angel
y=data["seed"].values.reshape(-1,1)
x=data["angel"].values.reshape(-1,1)
​
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
predictions = logmodel.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))
              precision    recall  f1-score   support

           0       0.72      1.00      0.84     11790
         118       0.00      0.00      0.00         1
         150       0.00      0.00      0.00         1
         929       0.00      0.00      0.00         1
        1000       0.00      0.00      0.00        11
        1506       0.00      0.00      0.00         1
        2000       0.00      0.00      0.00         2
        2500       0.00      0.00      0.00         2
        3000       0.00      0.00      0.00         3
        3345       0.00      0.00      0.00         1
        3766       0.00      0.00      0.00         1
        4000       0.00      0.00      0.00         2
        5000       0.00      0.00      0.00        14
        5001       0.00      0.00      0.00         1
        5005       0.00      0.00      0.00         1
        6000       0.00      0.00      0.00         1
        6500       0.00      0.00      0.00         1
        6597       0.00      0.00      0.00         1
        7500       0.00      0.00      0.00         4
        8024       0.00      0.00      0.00         1
        8595       0.00      0.00      0.00         1
        9000       0.00      0.00      0.00         2
        9542       0.00      0.00      0.00         1
        9677       0.00      0.00      0.00         1
        9791       0.00      0.00      0.00         1
       10000       0.00      0.00      0.00        59
       10240       0.00      0.00      0.00         1
       11308       0.00      0.00      0.00         1
       11347       0.00      0.00      0.00         1
       11650       0.00      0.00      0.00         1
       11700       0.00      0.00      0.00         2
       12000       0.00      0.00      0.00        12
       12273       0.00      0.00      0.00         1
       12500       0.00      0.00      0.00        17
       13000       0.00      0.00      0.00         1
       13259       0.00      0.00      0.00         1
       13503       0.00      0.00      0.00         1
       13510       0.00      0.00      0.00         1
       13596       0.00      0.00      0.00         1
       13597       0.00      0.00      0.00         1
       14000       0.00      0.00      0.00         3
       15000       0.00      0.00      0.00        48
       15270       0.00      0.00      0.00         1
       15873       0.00      0.00      0.00         1
       16000       0.00      0.00      0.00         4
       16567       0.00      0.00      0.00         1
       16574       0.00      0.00      0.00         1
       16600       0.00      0.00      0.00         1
       16645       0.00      0.00      0.00         1
       16800       0.00      0.00      0.00         1
       16977       0.00      0.00      0.00         1
       17000       0.00      0.00      0.00        10
       17191       0.00      0.00      0.00         1
       17408       0.00      0.00      0.00         1
       17500       0.00      0.00      0.00         1
       17966       0.00      0.00      0.00         1
       18000       0.00      0.00      0.00        12
       18316       0.00      0.00      0.00         1
       18357       0.00      0.00      0.00         1
       18584       0.00      0.00      0.00         1
       18852       0.00      0.00      0.00         4
       18885       0.00      0.00      0.00         3
       19000       0.00      0.00      0.00         1
       19252       0.00      0.00      0.00         3
       19299       0.00      0.00      0.00         8
       19311       0.00      0.00      0.00         2
       19654       0.00      0.00      0.00         1
       19713       0.00      0.00      0.00         1
       20000       0.00      0.00      0.00       102
       20014       0.00      0.00      0.00         1
       20118       0.00      0.00      0.00         4
       20203       0.00      0.00      0.00         5
       20265       0.00      0.00      0.00         1
       20313       0.00      0.00      0.00         1
       20352       0.00      0.00      0.00         1
       20391       0.00      0.00      0.00         1
       20395       0.00      0.00      0.00         1
       20962       0.00      0.00      0.00         1
       21000       0.00      0.00      0.00         2
       21189       0.00      0.00      0.00         4
       22000       0.00      0.00      0.00         2
       22091       0.00      0.00      0.00         2
       22500       0.00      0.00      0.00         2
       22522       0.00      0.00      0.00         1
       22818       0.00      0.00      0.00         4
       23911       0.00      0.00      0.00         1
       24000       0.00      0.00      0.00         1
       24148       0.00      0.00      0.00         1
       24329       0.00      0.00      0.00         1
       24844       0.00      0.00      0.00         1
       25000       0.00      0.00      0.00       110
       25025       0.00      0.00      0.00         3
       26000       0.00      0.00      0.00         1
       27000       0.00      0.00      0.00         3
       27034       0.00      0.00      0.00         2
       27181       0.00      0.00      0.00         1
       27193       0.00      0.00      0.00         1
       28000       0.00      0.00      0.00        13
       28437       0.00      0.00      0.00         2
       29222       0.00      0.00      0.00         1
       29358       0.00      0.00      0.00         4
       29411       0.00      0.00      0.00         1
       29600       0.00      0.00      0.00         1
       29651       0.00      0.00      0.00         1
       29833       0.00      0.00      0.00         1
       30000       0.00      0.00      0.00        38
       30965       0.00      0.00      0.00         1
       31500       0.00      0.00      0.00         1
       31520       0.00      0.00      0.00         1
       32000       0.00      0.00      0.00         1
       32165       0.00      0.00      0.00         3
       32500       0.00      0.00      0.00         1
       32680       0.00      0.00      0.00         1
       32811       0.00      0.00      0.00         1
       33000       0.00      0.00      0.00         1
       33078       0.00      0.00      0.00         1
       33390       0.00      0.00      0.00         1
       33457       0.00      0.00      0.00         1
       33724       0.00      0.00      0.00         1
       33793       0.00      0.00      0.00         2
       34400       0.00      0.00      0.00         1
       34513       0.00      0.00      0.00         1
       34623       0.00      0.00      0.00         1
       35000       0.00      0.00      0.00        13
       36000       0.00      0.00      0.00         1
       36154       0.00      0.00      0.00         1
       36198       0.00      0.00      0.00         1
       36570       0.00      0.00      0.00         1
       36802       0.00      0.00      0.00         1
       36967       0.00      0.00      0.00         1
       37000       0.00      0.00      0.00         2
       37397       0.00      0.00      0.00         1
       38000       0.00      0.00      0.00         1
       38484       0.00      0.00      0.00         1
       38495       0.00      0.00      0.00         1
       38598       0.00      0.00      0.00         1
       38817       0.00      0.00      0.00         1
       38979       0.00      0.00      0.00         1
       39080       0.00      0.00      0.00         1
       39083       0.00      0.00      0.00         1
       39959       0.00      0.00      0.00         1
       40000       0.00      0.00      0.00       223
       40331       0.00      0.00      0.00         1
       40404       0.00      0.00      0.00         1
       40635       0.00      0.00      0.00         1
       40800       0.00      0.00      0.00         1
       41004       0.00      0.00      0.00         1
       41069       0.00      0.00      0.00         1
       41250       0.00      0.00      0.00        13
       41500       0.00      0.00      0.00         2
       42000       0.00      0.00      0.00         1
       42183       0.00      0.00      0.00         1
       42500       0.00      0.00      0.00         1
       43358       0.00      0.00      0.00         1
       43833       0.00      0.00      0.00         1
       43859       0.00      0.00      0.00         1
       44000       0.00      0.00      0.00         1
       44496       0.00      0.00      0.00         1
       44843       0.00      0.00      0.00         1
       45000       0.00      0.00      0.00         7
       45062       0.00      0.00      0.00         1
       45091       0.00      0.00      0.00         1
       45112       0.00      0.00      0.00         1
       46500       0.00      0.00      0.00         1
       46549       0.00      0.00      0.00         1
       46599       0.00      0.00      0.00         2
       47000       0.00      0.00      0.00         1
       47168       0.00      0.00      0.00         1
       47226       0.00      0.00      0.00         1
       47413       0.00      0.00      0.00         1
       47500       0.00      0.00      0.00         1
       47917       0.00      0.00      0.00         2
       48000       0.00      0.00      0.00         2
       48049       0.00      0.00      0.00         1
       48632       0.00      0.00      0.00         1
       48701       0.00      0.00      0.00         1
       49224       0.00      0.00      0.00         2
       49247       0.00      0.00      0.00         1
       49530       0.00      0.00      0.00         1
       49618       0.00      0.00      0.00         1
       49682       0.00      0.00      0.00         2
       49805       0.00      0.00      0.00         1
       49837       0.00      0.00      0.00         1
       49877       0.00      0.00      0.00         1
       50000       0.00      0.00      0.00       126
       50051       0.00      0.00      0.00         1
       50116       0.00      0.00      0.00         1
       50301       0.00      0.00      0.00         1
       50311       0.00      0.00      0.00         1
       50458       0.00      0.00      0.00         1
       50695       0.00      0.00      0.00         1
       51022       0.00      0.00      0.00         2
       51133       0.00      0.00      0.00         1
       51220       0.00      0.00      0.00         2
       51250       0.00      0.00      0.00         1
       51295       0.00      0.00      0.00         3
       51713       0.00      0.00      0.00         1
       51728       0.00      0.00      0.00         1
       51842       0.00      0.00      0.00         1
       52000       0.00      0.00      0.00         2
       52003       0.00      0.00      0.00         1
       52034       0.00      0.00      0.00         1
       52122       0.00      0.00      0.00         1
       52430       0.00      0.00      0.00         1
       52500       0.00      0.00      0.00         1
       52792       0.00      0.00      0.00         1
       52869       0.00      0.00      0.00         1
       53000       0.00      0.00      0.00         1
       53016       0.00      0.00      0.00         2
       53102       0.00      0.00      0.00         1
       53159       0.00      0.00      0.00         1
       53373       0.00      0.00      0.00         1
       53448       0.00      0.00      0.00         1
       53491       0.00      0.00      0.00         1
       53750       0.00      0.00      0.00         1
       53968       0.00      0.00      0.00         1
       54000       0.00      0.00      0.00         1
       54068       0.00      0.00      0.00         1
       54500       0.00      0.00      0.00         1
       54908       0.00      0.00      0.00         1
       55000       0.00      0.00      0.00         7
       55500       0.00      0.00      0.00         1
       56737       0.00      0.00      0.00         1
       56874       0.00      0.00      0.00         1
       56900       0.00      0.00      0.00         1
       57000       0.00      0.00      0.00         1
       57166       0.00      0.00      0.00         1
       57312       0.00      0.00      0.00         1
       58000       0.00      0.00      0.00         1
       58438       0.00      0.00      0.00         1
       59145       0.00      0.00      0.00         1
       59376       0.00      0.00      0.00         1
       59390       0.00      0.00      0.00         1
       60000       0.00      0.00      0.00        28
       60532       0.00      0.00      0.00         1
       62013       0.00      0.00      0.00         1
       62500       0.00      0.00      0.00         2
       62607       0.00      0.00      0.00         1
       63276       0.00      0.00      0.00         1
       63398       0.00      0.00      0.00         1
       63925       0.00      0.00      0.00         1
       64000       0.00      0.00      0.00         1
       64021       0.00      0.00      0.00         1
       64300       0.00      0.00      0.00         1
       64330       0.00      0.00      0.00         4
       64437       0.00      0.00      0.00         1
       64500       0.00      0.00      0.00         1
       64525       0.00      0.00      0.00         1
       64602       0.00      0.00      0.00         1
       64630       0.00      0.00      0.00         2
       64845       0.00      0.00      0.00         2
       65000       0.00      0.00      0.00        12
       65482       0.00      0.00      0.00         1
       65665       0.00      0.00      0.00         1
       65684       0.00      0.00      0.00         1
       65952       0.00      0.00      0.00         1
       65970       0.00      0.00      0.00         1
       66023       0.00      0.00      0.00         1
       66033       0.00      0.00      0.00         1
       66047       0.00      0.00      0.00         2
       66070       0.00      0.00      0.00         1
       66299       0.00      0.00      0.00         1
       66372       0.00      0.00      0.00         1
       66593       0.00      0.00      0.00         1
       66674       0.00      0.00      0.00         1
       66891       0.00      0.00      0.00         1
       67478       0.00      0.00      0.00         1
       67530       0.00      0.00      0.00         1
       67567       0.00      0.00      0.00         1
       67631       0.00      0.00      0.00         1
       68000       0.00      0.00      0.00         1
       68142       0.00      0.00      0.00         1
       68195       0.00      0.00      0.00         1
       68525       0.00      0.00      0.00         1
       68536       0.00      0.00      0.00         1
       68839       0.00      0.00      0.00         1
       68935       0.00      0.00      0.00         1
       69000       0.00      0.00      0.00         1
       69042       0.00      0.00      0.00         1
       69247       0.00      0.00      0.00         2
       69585       0.00      0.00      0.00         1
       70000       0.00      0.00      0.00        14
       70020       0.00      0.00      0.00         1
       70052       0.00      0.00      0.00         1
       71000       0.00      0.00      0.00         1
       71362       0.00      0.00      0.00         1
       71942       0.00      0.00      0.00         1
       72000       0.00      0.00      0.00         1
       72500       0.00      0.00      0.00         1
       73000       0.00      0.00      0.00         1
       73055       0.00      0.00      0.00         1
       73282       0.00      0.00      0.00         1
       74897       0.00      0.00      0.00         1
       75000       0.00      0.00      0.00        27
       75027       0.00      0.00      0.00         1
       75336       0.00      0.00      0.00         1
       75762       0.00      0.00      0.00         2
       76130       0.00      0.00      0.00         1
       76269       0.00      0.00      0.00         1
       76500       0.00      0.00      0.00         1
       76800       0.00      0.00      0.00         1
       78360       0.00      0.00      0.00         1
       78820       0.00      0.00      0.00         1
       79189       0.00      0.00      0.00         1
       79248       0.00      0.00      0.00         1
       79725       0.00      0.00      0.00         1
       80000       0.00      0.00      0.00        11
       80069       0.00      0.00      0.00         1
       80153       0.00      0.00      0.00         1
       80752       0.00      0.00      0.00         1
       80953       0.00      0.00      0.00         1
       81208       0.00      0.00      0.00         1
       81328       0.00      0.00      0.00         1
       81845       0.00      0.00      0.00         1
       81874       0.00      0.00      0.00         1
       82695       0.00      0.00      0.00         2
       83053       0.00      0.00      0.00         1
       84655       0.00      0.00      0.00         1
       85000       0.00      0.00      0.00         4
       85761       0.00      0.00      0.00         1
       86281       0.00      0.00      0.00         1
       87000       0.00      0.00      0.00         3
       87468       0.00      0.00      0.00         1
       88049       0.00      0.00      0.00         1
       89998       0.00      0.00      0.00         1
       90000       0.00      0.00      0.00        11
       90062       0.00      0.00      0.00         1
       91500       0.00      0.00      0.00         1
       92557       0.00      0.00      0.00         1
       92862       0.00      0.00      0.00         1
       94469       0.00      0.00      0.00         1
       94582       0.00      0.00      0.00         1
       94599       0.00      0.00      0.00         1
       94637       0.00      0.00      0.00         1
       94759       0.00      0.00      0.00         1
       95000       0.00      0.00      0.00         3
       95389       0.00      0.00      0.00         1
       96984       0.00      0.00      0.00         1
       98000       0.00      0.00      0.00         1
       99627       0.00      0.00      0.00         1
      100000       0.00      0.00      0.00       186
      100320       0.00      0.00      0.00         1
      100793       0.00      0.00      0.00         1
      101688       0.00      0.00      0.00         1
      102809       0.00      0.00      0.00         1
      103403       0.00      0.00      0.00         1
      103465       0.00      0.00      0.00         1
      104000       0.00      0.00      0.00         1
      104978       0.00      0.00      0.00         1
      105000       0.00      0.00      0.00         2
      106000       0.00      0.00      0.00         1
      109518       0.00      0.00      0.00         1
      110000       0.00      0.00      0.00        13
      110393       0.00      0.00      0.00         1
      111000       0.00      0.00      0.00         1
      111540       0.00      0.00      0.00         1
      112000       0.00      0.00      0.00         2
      112500       0.00      0.00      0.00         1
      112946       0.00      0.00      0.00         1
      113351       0.00      0.00      0.00         1
      114000       0.00      0.00      0.00         1
      115000       0.00      0.00      0.00         3
      115015       0.00      0.00      0.00         1
      117059       0.00      0.00      0.00         1
      117266       0.00      0.00      0.00         1
      118000       0.00      0.00      0.00        13
      118400       0.00      0.00      0.00         1
      118956       0.00      0.00      0.00         1
      119066       0.00      0.00      0.00         1
      119916       0.00      0.00      0.00         1
      119998       0.00      0.00      0.00         1
      120000       0.00      0.00      0.00        34
      120661       0.00      0.00      0.00         1
      120973       0.00      0.00      0.00         1
      121700       0.00      0.00      0.00         1
      121753       0.00      0.00      0.00         1
      121857       0.00      0.00      0.00         1
      122000       0.00      0.00      0.00         2
      122408       0.00      0.00      0.00         1
      122476       0.00      0.00      0.00         1
      123045       0.00      0.00      0.00         1
      124143       0.00      0.00      0.00         1
      124690       0.00      0.00      0.00         1
      125000       0.00      0.00      0.00        17
      125776       0.00      0.00      0.00         1
      126180       0.00      0.00      0.00         1
      126814       0.00      0.00      0.00         1
      126857       0.00      0.00      0.00         1
      127450       0.00      0.00      0.00         1
      127469       0.00      0.00      0.00         1
      127527       0.00      0.00      0.00         1
      127619       0.00      0.00      0.00         1
      128660       0.00      0.00      0.00         2
      129278       0.00      0.00      0.00         1
      129390       0.00      0.00      0.00         1
      129679       0.00      0.00      0.00         1
      130000       0.00      0.00      0.00         7
      130390       0.00      0.00      0.00         1
      131529       0.00      0.00      0.00         1
      131700       0.00      0.00      0.00         1
      131945       0.00      0.00      0.00         1
      132047       0.00      0.00      0.00         1
      132750       0.00      0.00      0.00         1
      132975       0.00      0.00      0.00         1
      133150       0.00      0.00      0.00         1
      133560       0.00      0.00      0.00         1
      134649       0.00      0.00      0.00         1
      134935       0.00      0.00      0.00         1
      135000       0.00      0.00      0.00         4
      135377       0.00      0.00      0.00         1
      135753       0.00      0.00      0.00         1
      135952       0.00      0.00      0.00         1
      136138       0.00      0.00      0.00         1
      136270       0.00      0.00      0.00         1
      136843       0.00      0.00      0.00         1
      137000       0.00      0.00      0.00         1
      137050       0.00      0.00      0.00         1
      137104       0.00      0.00      0.00         1
      139787       0.00      0.00      0.00         1
      140000       0.00      0.00      0.00        15
      140496       0.00      0.00      0.00         1
      143079       0.00      0.00      0.00         1
      144000       0.00      0.00      0.00         1
      144424       0.00      0.00      0.00         1
      144814       0.00      0.00      0.00         1
      145380       0.00      0.00      0.00         1
      145390       0.00      0.00      0.00         1
      145563       0.00      0.00      0.00         1
      146957       0.00      0.00      0.00         1
      147159       0.00      0.00      0.00         1
      148323       0.00      0.00      0.00         1
      149000       0.00      0.00      0.00         1
      149996       0.00      0.00      0.00         1
      150000       0.00      0.00      0.00        84
      150040       0.00      0.00      0.00         1
      150450       0.00      0.00      0.00         1
      151022       0.00      0.00      0.00         1
      153317       0.00      0.00      0.00         1
      154392       0.00      0.00      0.00         1
      155000       0.00      0.00      0.00         2
      155268       0.00      0.00      0.00         1
      156286       0.00      0.00      0.00         1
      157896       0.00      0.00      0.00         1
      158000       0.00      0.00      0.00         2
      158567       0.00      0.00      0.00         1
      159000       0.00      0.00      0.00         1
      160000       0.00      0.00      0.00        10
      161391       0.00      0.00      0.00         1
      161671       0.00      0.00      0.00         1
      161779       0.00      0.00      0.00         1
      162107       0.00      0.00      0.00         1
      162778       0.00      0.00      0.00         1
      163000       0.00      0.00      0.00         2
      163391       0.00      0.00      0.00         1
      164223       0.00      0.00      0.00         1
      165000       0.00      0.00      0.00         1
      165543       0.00      0.00      0.00         1
      166455       0.00      0.00      0.00         1
      166677       0.00      0.00      0.00         1
      166975       0.00      0.00      0.00         1
      167000       0.00      0.00      0.00         1
      167258       0.00      0.00      0.00         1
      168403       0.00      0.00      0.00         1
      170000       0.00      0.00      0.00        11
      171942       0.00      0.00      0.00         1
      172801       0.00      0.00      0.00         1
      174390       0.00      0.00      0.00         1
      175000       0.00      0.00      0.00         6
      176800       0.00      0.00      0.00         1
      178486       0.00      0.00      0.00         1
      179232       0.00      0.00      0.00         1
      180000       0.00      0.00      0.00         7
      182324       0.00      0.00      0.00         1
      183747       0.00      0.00      0.00         1
      184000       0.00      0.00      0.00         1
      184103       0.00      0.00      0.00         1
      184711       0.00      0.00      0.00         1
      185000       0.00      0.00      0.00         1
      186500       0.00      0.00      0.00         1
      187000       0.00      0.00      0.00         1
      187632       0.00      0.00      0.00         1
      188642       0.00      0.00      0.00         1
      189406       0.00      0.00      0.00         1
      190000       0.00      0.00      0.00         1
      190035       0.00      0.00      0.00         1
      190466       0.00      0.00      0.00         1
      191876       0.00      0.00      0.00         1
      192000       0.00      0.00      0.00         1
      192621       0.00      0.00      0.00         1
      192990       0.00      0.00      0.00         1
      193000       0.00      0.00      0.00         1
      195000       0.00      0.00      0.00         1
      195420       0.00      0.00      0.00         1
      195607       0.00      0.00      0.00         1
      196079       0.00      0.00      0.00         1
      197550       0.00      0.00      0.00         1
      197808       0.00      0.00      0.00         1
      199816       0.00      0.00      0.00         1
      200000       0.00      0.00      0.00        83
      200002       0.00      0.00      0.00         1
      200010       0.00      0.00      0.00         1
      200100       0.00      0.00      0.00         1
      200212       0.00      0.00      0.00         1
      200478       0.00      0.00      0.00         1
      201000       0.00      0.00      0.00         1
      201956       0.00      0.00      0.00         1
      204000       0.00      0.00      0.00         1
      204923       0.00      0.00      0.00         1
      205000       0.00      0.00      0.00         3
      208354       0.00      0.00      0.00         1
      210000       0.00      0.00      0.00         5
      212361       0.00      0.00      0.00         1
      212697       0.00      0.00      0.00         1
      213300       0.00      0.00      0.00         1
      215000       0.00      0.00      0.00         2
      219165       0.00      0.00      0.00         1
      219795       0.00      0.00      0.00         1
      220000       0.00      0.00      0.00         4
      225000       0.00      0.00      0.00        11
      225500       0.00      0.00      0.00         1
      226110       0.00      0.00      0.00         1
      226331       0.00      0.00      0.00         1
      227287       0.00      0.00      0.00         1
      227975       0.00      0.00      0.00         1
      228000       0.00      0.00      0.00         2
      228091       0.00      0.00      0.00         1
      228260       0.00      0.00      0.00         1
      230000       0.00      0.00      0.00         3
      230496       0.00      0.00      0.00         1
      230627       0.00      0.00      0.00         1
      231052       0.00      0.00      0.00         1
      231949       0.00      0.00      0.00         1
      232095       0.00      0.00      0.00         1
      232629       0.00      0.00      0.00         1
      233100       0.00      0.00      0.00         1
      233536       0.00      0.00      0.00         1
      234900       0.00      0.00      0.00         1
      235000       0.00      0.00      0.00         4
      237000       0.00      0.00      0.00         1
      240000       0.00      0.00      0.00         3
      242718       0.00      0.00      0.00         1
      247000       0.00      0.00      0.00         1
      248988       0.00      0.00      0.00         1
      249304       0.00      0.00      0.00         1
      250000       0.00      0.00      0.00        95
      250611       0.00      0.00      0.00         1
      252040       0.00      0.00      0.00         1
      253107       0.00      0.00      0.00         1
      254220       0.00      0.00      0.00         1
      255000       0.00      0.00      0.00         1
      255172       0.00      0.00      0.00         1
      257320       0.00      0.00      0.00         3
      257913       0.00      0.00      0.00         1
      258957       0.00      0.00      0.00         1
      260000       0.00      0.00      0.00         4
      261360       0.00      0.00      0.00         1
      262500       0.00      0.00      0.00         1
      265000       0.00      0.00      0.00         4
      266666       0.00      0.00      0.00         1
      268000       0.00      0.00      0.00         1
      268791       0.00      0.00      0.00         1
      269260       0.00      0.00      0.00         1
      270000       0.00      0.00      0.00         3
      271472       0.00      0.00      0.00         1
      273453       0.00      0.00      0.00         2
      275000       0.00      0.00      0.00         5
      275517       0.00      0.00      0.00         1
      275618       0.00      0.00      0.00         1
      276375       0.00      0.00      0.00         1
      278088       0.00      0.00      0.00         1
      278336       0.00      0.00      0.00         1
      279251       0.00      0.00      0.00         1
      279706       0.00      0.00      0.00         1
      280000       0.00      0.00      0.00         5
      280080       0.00      0.00      0.00         1
      280338       0.00      0.00      0.00         1
      282640       0.00      0.00      0.00         1
      284775       0.00      0.00      0.00         1
      285000       0.00      0.00      0.00         1
      286760       0.00      0.00      0.00         1
      288000       0.00      0.00      0.00         1
      288120       0.00      0.00      0.00         1
      289589       0.00      0.00      0.00         1
      290000       0.00      0.00      0.00         2
      292384       0.00      0.00      0.00         1
      295000       0.00      0.00      0.00         1
      297176       0.00      0.00      0.00         1
      300000       0.00      0.00      0.00        70
      303064       0.00      0.00      0.00         1
      303980       0.00      0.00      0.00         1
      307167       0.00      0.00      0.00         1
      308172       0.00      0.00      0.00         1
      310000       0.00      0.00      0.00         5
      313000       0.00      0.00      0.00         1
      315000       0.00      0.00      0.00         1
      315072       0.00      0.00      0.00         1
      315317       0.00      0.00      0.00         1
      315380       0.00      0.00      0.00         1
      320000       0.00      0.00      0.00         3
      321650       0.00      0.00      0.00         1
      322500       0.00      0.00      0.00         1
      322917       0.00      0.00      0.00         1
      325000       0.00      0.00      0.00         8
      326647       0.00      0.00      0.00         1
      327352       0.00      0.00      0.00         1
      327774       0.00      0.00      0.00         1
      330000       0.00      0.00      0.00         4
      330813       0.00      0.00      0.00         1
      333333       0.00      0.00      0.00         1
      334260       0.00      0.00      0.00         1
      335000       0.00      0.00      0.00         4
      340170       0.00      0.00      0.00         1
      340497       0.00      0.00      0.00         1
      340525       0.00      0.00      0.00         2
      342447       0.00      0.00      0.00         1
      348253       0.00      0.00      0.00         1
      349999       0.00      0.00      0.00         1
      350000       0.00      0.00      0.00        32
      350363       0.00      0.00      0.00         1
      350721       0.00      0.00      0.00         1
      355000       0.00      0.00      0.00         2
      357000       0.00      0.00      0.00         1
      357850       0.00      0.00      0.00         1
      360000       0.00      0.00      0.00         3
      362240       0.00      0.00      0.00         1
      363385       0.00      0.00      0.00         1
      365000       0.00      0.00      0.00         5
      369311       0.00      0.00      0.00         1
      369713       0.00      0.00      0.00         1
      370000       0.00      0.00      0.00         1
      370207       0.00      0.00      0.00         1
      371696       0.00      0.00      0.00         1
      372568       0.00      0.00      0.00         1
      372752       0.00      0.00      0.00         1
      374999       0.00      0.00      0.00         1
      375000       0.00      0.00      0.00         8
      375575       0.00      0.00      0.00         1
      376256       0.00      0.00      0.00         1
      376917       0.00      0.00      0.00         1
      378000       0.00      0.00      0.00         1
      379147       0.00      0.00      0.00         1
      380000       0.00      0.00      0.00         3
      380110       0.00      0.00      0.00         1
      382096       0.00      0.00      0.00         1
      384879       0.00      0.00      0.00         1
      385000       0.00      0.00      0.00         2
      385980       0.00      0.00      0.00         2
      387000       0.00      0.00      0.00         1
      388050       0.00      0.00      0.00         1
      389454       0.00      0.00      0.00         1
      390000       0.00      0.00      0.00         3
      390360       0.00      0.00      0.00         2
      392500       0.00      0.00      0.00         1
      392584       0.00      0.00      0.00         1
      394104       0.00      0.00      0.00         1
      395000       0.00      0.00      0.00         1
      400000       0.00      0.00      0.00        43
      402500       0.00      0.00      0.00         1
      405002       0.00      0.00      0.00         1
      405810       0.00      0.00      0.00         1
      409268       0.00      0.00      0.00         1
      410000       0.00      0.00      0.00         2
      410076       0.00      0.00      0.00         1
      412000       0.00      0.00      0.00         1
      413612       0.00      0.00      0.00         2
      415008       0.00      0.00      0.00         1
      416139       0.00      0.00      0.00         1
      417000       0.00      0.00      0.00         1
      417212       0.00      0.00      0.00         1
      417780       0.00      0.00      0.00         1
      419047       0.00      0.00      0.00         1
      419688       0.00      0.00      0.00         1
      420000       0.00      0.00      0.00         4
      421326       0.00      0.00      0.00         1
      425000       0.00      0.00      0.00         4
      425975       0.00      0.00      0.00         1
      428257       0.00      0.00      0.00         1
      430139       0.00      0.00      0.00         1
      430855       0.00      0.00      0.00         1
      435000       0.00      0.00      0.00         2
      440000       0.00      0.00      0.00         3
      442000       0.00      0.00      0.00         1
      445000       0.00      0.00      0.00         1
      446995       0.00      0.00      0.00         1
      448000       0.00      0.00      0.00         1
      448615       0.00      0.00      0.00         1
      450000       0.00      0.00      0.00        10
      450240       0.00      0.00      0.00         1
      450450       0.00      0.00      0.00         1
      452550       0.00      0.00      0.00         1
      454545       0.00      0.00      0.00         1
      455000       0.00      0.00      0.00         1
      455970       0.00      0.00      0.00         1
      457590       0.00      0.00      0.00         1
      459937       0.00      0.00      0.00         1
      460000       0.00      0.00      0.00         1
      461500       0.00      0.00      0.00         1
      465850       0.00      0.00      0.00         1
      468000       0.00      0.00      0.00         1
      469673       0.00      0.00      0.00         1
      470000       0.00      0.00      0.00         1
      471000       0.00      0.00      0.00         1
      474545       0.00      0.00      0.00         1
      475000       0.00      0.00      0.00         2
      477181       0.00      0.00      0.00         1
      479233       0.00      0.00      0.00         1
      479998       0.00      0.00      0.00         1
      480000       0.00      0.00      0.00         2
      480324       0.00      0.00      0.00         1
      483333       0.00      0.00      0.00         1
      484900       0.00      0.00      0.00         1
      485000       0.00      0.00      0.00         3
      486142       0.00      0.00      0.00         1
      488140       0.00      0.00      0.00         1
      490000       0.00      0.00      0.00         1
      490388       0.00      0.00      0.00         1
      494833       0.00      0.00      0.00         1
      495675       0.00      0.00      0.00         1
      497388       0.00      0.00      0.00         1
      500000       0.00      0.00      0.00       179
      500184       0.00      0.00      0.00         1
      501408       0.00      0.00      0.00         1
      501810       0.00      0.00      0.00         1
      502500       0.00      0.00      0.00         1
      503176       0.00      0.00      0.00         1
      504512       0.00      0.00      0.00         1
      505000       0.00      0.00      0.00         3
      510000       0.00      0.00      0.00         1
      510294       0.00      0.00      0.00         1
      512000       0.00      0.00      0.00         1
      512340       0.00      0.00      0.00         1
      513200       0.00      0.00      0.00         1
      516000       0.00      0.00      0.00         1
      518000       0.00      0.00      0.00         1
      520000       0.00      0.00      0.00         1
      525000       0.00      0.00      0.00         2
      525457       0.00      0.00      0.00         1
      528000       0.00      0.00      0.00         1
      530000       0.00      0.00      0.00         4
      535000       0.00      0.00      0.00         2
      535144       0.00      0.00      0.00         1
      535952       0.00      0.00      0.00         1
      537868       0.00      0.00      0.00         1
      538062       0.00      0.00      0.00         1
      540000       0.00      0.00      0.00         6
      540010       0.00      0.00      0.00         1
      541250       0.00      0.00      0.00         1
      545000       0.00      0.00      0.00         2
      546000       0.00      0.00      0.00         1
      550000       0.00      0.00      0.00        21
      550994       0.00      0.00      0.00         1
      555026       0.00      0.00      0.00         1
      558612       0.00      0.00      0.00         1
      559584       0.00      0.00      0.00         1
      559992       0.00      0.00      0.00         1
      560000       0.00      0.00      0.00         5
      560760       0.00      0.00      0.00         1
      561238       0.00      0.00      0.00         1
      562000       0.00      0.00      0.00         1
      565000       0.00      0.00      0.00         2
      569518       0.00      0.00      0.00         1
      570076       0.00      0.00      0.00         1
      570966       0.00      0.00      0.00         1
      572670       0.00      0.00      0.00         1
      573835       0.00      0.00      0.00         1
      575000       0.00      0.00      0.00         2
      575002       0.00      0.00      0.00         1
      575608       0.00      0.00      0.00         1
      576456       0.00      0.00      0.00         1
      580000       0.00      0.00      0.00         1
      585000       0.00      0.00      0.00         1
      585191       0.00      0.00      0.00         1
      588472       0.00      0.00      0.00         1
      588840       0.00      0.00      0.00         1
      592428       0.00      0.00      0.00         1
      595000       0.00      0.00      0.00         1
      597001       0.00      0.00      0.00         1
      599205       0.00      0.00      0.00         1
      600000       0.00      0.00      0.00        57
      600015       0.00      0.00      0.00         1
      600314       0.00      0.00      0.00         1
      600364       0.00      0.00      0.00         1
      606672       0.00      0.00      0.00         1
      607200       0.00      0.00      0.00         1
      610000       0.00      0.00      0.00         3
      613791       0.00      0.00      0.00         1
      614900       0.00      0.00      0.00         1
      617000       0.00      0.00      0.00         1
      618000       0.00      0.00      0.00         2
      619060       0.00      0.00      0.00         1
      620000       0.00      0.00      0.00         2
      621995       0.00      0.00      0.00         1
      622125       0.00      0.00      0.00         1
      623500       0.00      0.00      0.00         1
      625000       0.00      0.00      0.00         2
      631700       0.00      0.00      0.00         1
      635820       0.00      0.00      0.00         1
      637850       0.00      0.00      0.00         1
      640000       0.00      0.00      0.00         2
      642000       0.00      0.00      0.00         1
      643300       0.00      0.00      0.00         2
      645088       0.00      0.00      0.00         1
      646774       0.00      0.00      0.00         1
      646950       0.00      0.00      0.00         1
      648414       0.00      0.00      0.00         1
      650000       0.00      0.00      0.00        26
      651339       0.00      0.00      0.00         1
      653000       0.00      0.00      0.00         1
      654553       0.00      0.00      0.00         1
      654957       0.00      0.00      0.00         1
      656598       0.00      0.00      0.00         1
      659134       0.00      0.00      0.00         1
      662152       0.00      0.00      0.00         1
      663800       0.00      0.00      0.00         1
      665000       0.00      0.00      0.00         1
      665750       0.00      0.00      0.00         1
      669032       0.00      0.00      0.00         1
      670000       0.00      0.00      0.00         1
      675000       0.00      0.00      0.00         2
      676000       0.00      0.00      0.00         1
      676644       0.00      0.00      0.00         1
      685000       0.00      0.00      0.00         3
      685166       0.00      0.00      0.00         1
      686000       0.00      0.00      0.00         1
      687000       0.00      0.00      0.00         1
      692635       0.00      0.00      0.00         1
      698453       0.00      0.00      0.00         1
      700000       0.00      0.00      0.00        37
      703883       0.00      0.00      0.00         1
      705000       0.00      0.00      0.00         1
      710000       0.00      0.00      0.00         1
      715000       0.00      0.00      0.00         1
      718000       0.00      0.00      0.00         1
      718350       0.00      0.00      0.00         1
      718450       0.00      0.00      0.00         1
      720000       0.00      0.00      0.00         1
      720496       0.00      0.00      0.00         1
      722455       0.00      0.00      0.00         1
      724071       0.00      0.00      0.00         1
      725000       0.00      0.00      0.00         2
      730000       0.00      0.00      0.00         2
      732800       0.00      0.00      0.00         1
      734828       0.00      0.00      0.00         1
      735480       0.00      0.00      0.00         1
      736500       0.00      0.00      0.00         1
      740000       0.00      0.00      0.00         2
      740427       0.00      0.00      0.00         1
      741017       0.00      0.00      0.00         1
      742000       0.00      0.00      0.00         1
      745000       0.00      0.00      0.00         3
      750000       0.00      0.00      0.00        44
      752161       0.00      0.00      0.00         1
      754466       0.00      0.00      0.00         1
      757625       0.00      0.00      0.00         2
      759146       0.00      0.00      0.00         1
      759840       0.00      0.00      0.00         1
      762000       0.00      0.00      0.00         1
      762927       0.00      0.00      0.00         1
      765000       0.00      0.00      0.00         1
      767473       0.00      0.00      0.00         1
      770000       0.00      0.00      0.00         4
      771960       0.00      0.00      0.00         1
      775000       0.00      0.00      0.00         1
      775838       0.00      0.00      0.00         1
      777338       0.00      0.00      0.00         1
      778000       0.00      0.00      0.00         1
      778110       0.00      0.00      0.00         1
      778143       0.00      0.00      0.00         1
      779689       0.00      0.00      0.00         1
      780000       0.00      0.00      0.00         1
      780120       0.00      0.00      0.00         1
      780511       0.00      0.00      0.00         1
      780646       0.00      0.00      0.00         1
      784980       0.00      0.00      0.00         1
      785000       0.00      0.00      0.00         1
      786993       0.00      0.00      0.00         1
      792506       0.00      0.00      0.00         1
      794760       0.00      0.00      0.00         1
      795000       0.00      0.00      0.00         1
      795500       0.00      0.00      0.00         1
      796760       0.00      0.00      0.00         1
      797790       0.00      0.00      0.00         1
      798887       0.00      0.00      0.00         1
      800000       0.00      0.00      0.00        36
      800924       0.00      0.00      0.00         1
      800958       0.00      0.00      0.00         1
      801929       0.00      0.00      0.00         1
      802500       0.00      0.00      0.00         1
      808211       0.00      0.00      0.00         1
      811057       0.00      0.00      0.00         1
      813241       0.00      0.00      0.00         1
      820000       0.00      0.00      0.00         1
      825000       0.00      0.00      0.00         2
      826355       0.00      0.00      0.00         1
      830000       0.00      0.00      0.00         1
      831000       0.00      0.00      0.00         1
      833333       0.00      0.00      0.00         1
      840000       0.00      0.00      0.00         1
      842723       0.00      0.00      0.00         1
      843150       0.00      0.00      0.00         1
      847104       0.00      0.00      0.00         1
      850000       0.00      0.00      0.00        14
      857480       0.00      0.00      0.00         1
      857767       0.00      0.00      0.00         1
      858500       0.00      0.00      0.00         1
      860000       0.00      0.00      0.00         1
      863400       0.00      0.00      0.00         1
      865000       0.00      0.00      0.00         1
      868000       0.00      0.00      0.00         2
      870000       0.00      0.00      0.00         2
      871000       0.00      0.00      0.00         1
      875000       0.00      0.00      0.00         3
      878619       0.00      0.00      0.00         1
      879531       0.00      0.00      0.00         1
      880000       0.00      0.00      0.00         2
      881025       0.00      0.00      0.00         1
      887000       0.00      0.00      0.00         1
      893679       0.00      0.00      0.00         1
      899955       0.00      0.00      0.00         1
      900000       0.00      0.00      0.00        14
      900001       0.00      0.00      0.00         1
      900620       0.00      0.00      0.00         1
      904725       0.00      0.00      0.00         1
      908000       0.00      0.00      0.00         1
      909150       0.00      0.00      0.00         1
      914005       0.00      0.00      0.00         1
      915000       0.00      0.00      0.00         1
      918741       0.00      0.00      0.00         1
      920000       0.00      0.00      0.00         2
      922350       0.00      0.00      0.00         1
      925000       0.00      0.00      0.00         1
      930000       0.00      0.00      0.00         2
      936000       0.00      0.00      0.00         1
      940000       0.00      0.00      0.00         1
      946000       0.00      0.00      0.00         1
      947228       0.00      0.00      0.00         1
      950000       0.00      0.00      0.00         5
      950563       0.00      0.00      0.00         1
      952380       0.00      0.00      0.00         1
      960470       0.00      0.00      0.00         1
      961000       0.00      0.00      0.00         1
      965754       0.00      0.00      0.00         1
      969000       0.00      0.00      0.00         1
      969533       0.00      0.00      0.00         1
      970000       0.00      0.00      0.00         1
      975000       0.00      0.00      0.00         2
      978626       0.00      0.00      0.00         1
      980000       0.00      0.00      0.00         2
      981217       0.00      0.00      0.00         1
      982000       0.00      0.00      0.00         1
      984000       0.00      0.00      0.00         1
      990000       0.00      0.00      0.00         1
      992250       0.00      0.00      0.00         1
      992514       0.00      0.00      0.00         1
      994000       0.00      0.00      0.00         1
      999984       0.00      0.00      0.00         1
      999999       0.00      0.00      0.00         1
     1000000       0.00      0.00      0.00       170
     1003681       0.00      0.00      0.00         1
     1010000       0.00      0.00      0.00         2
     1012000       0.00      0.00      0.00         1
     1015921       0.00      0.00      0.00         1
     1016000       0.00      0.00      0.00         1
     1020000       0.00      0.00      0.00         1
     1022917       0.00      0.00      0.00         1
     1024999       0.00      0.00      0.00         1
     1025000       0.00      0.00      0.00         2
     1026000       0.00      0.00      0.00         1
     1030000       0.00      0.00      0.00         1
     1038090       0.00      0.00      0.00         1
     1038955       0.00      0.00      0.00         1
     1040000       0.00      0.00      0.00         1
     1045607       0.00      0.00      0.00         1
     1049999       0.00      0.00      0.00         1
     1050000       0.00      0.00      0.00         5
     1054991       0.00      0.00      0.00         1
     1055000       0.00      0.00      0.00         1
     1060000       0.00      0.00      0.00         1
     1068000       0.00      0.00      0.00         1
     1068750       0.00      0.00      0.00         1
     1070000       0.00      0.00      0.00         1
     1075000       0.00      0.00      0.00         2
     1080001       0.00      0.00      0.00         1
     1080450       0.00      0.00      0.00         1
     1086016       0.00      0.00      0.00         1
     1091663       0.00      0.00      0.00         1
     1096318       0.00      0.00      0.00         1
     1098900       0.00      0.00      0.00         1
     1099309       0.00      0.00      0.00         1
     1099998       0.00      0.00      0.00         1
     1100000       0.00      0.00      0.00        28
     1106990       0.00      0.00      0.00         1
     1111351       0.00      0.00      0.00         1
     1125000       0.00      0.00      0.00         4
     1127730       0.00      0.00      0.00         1
     1134229       0.00      0.00      0.00         1
     1136438       0.00      0.00      0.00         1
     1140000       0.00      0.00      0.00         1
     1149000       0.00      0.00      0.00         1
     1150000       0.00      0.00      0.00         2
     1158685       0.00      0.00      0.00         1
     1164000       0.00      0.00      0.00         1
     1166220       0.00      0.00      0.00         1
     1168000       0.00      0.00      0.00         1
     1175000       0.00      0.00      0.00         2
     1180000       0.00      0.00      0.00         1
     1183160       0.00      0.00      0.00         1
     1185600       0.00      0.00      0.00         1
     1185800       0.00      0.00      0.00         1
     1189120       0.00      0.00      0.00         1
     1199998       0.00      0.00      0.00         1
     1200000       0.00      0.00      0.00        46
     1203068       0.00      0.00      0.00         1
     1205000       0.00      0.00      0.00         1
     1210000       0.00      0.00      0.00         1
     1212921       0.00      0.00      0.00         1
     1213000       0.00      0.00      0.00         1
     1213010       0.00      0.00      0.00         1
     1216100       0.00      0.00      0.00         1
     1217000       0.00      0.00      0.00         1
     1219452       0.00      0.00      0.00         1
     1220000       0.00      0.00      0.00         4
     1230000       0.00      0.00      0.00         1
     1231756       0.00      0.00      0.00         1
     1235000       0.00      0.00      0.00         1
     1249991       0.00      0.00      0.00         1
     1250000       0.00      0.00      0.00        28
     1251285       0.00      0.00      0.00         1
     1260000       0.00      0.00      0.00         1
     1264798       0.00      0.00      0.00         1
     1265000       0.00      0.00      0.00         1
     1275000       0.00      0.00      0.00         1
     1275490       0.00      0.00      0.00         1
     1278000       0.00      0.00      0.00         1
     1285997       0.00      0.00      0.00         1
     1286600       0.00      0.00      0.00         2
     1287243       0.00      0.00      0.00         1
     1287963       0.00      0.00      0.00         1
     1296500       0.00      0.00      0.00         1
     1300000       0.00      0.00      0.00        25
     1301000       0.00      0.00      0.00         1
     1303116       0.00      0.00      0.00         1
     1309106       0.00      0.00      0.00         1
     1320000       0.00      0.00      0.00         3
     1320495       0.00      0.00      0.00         1
     1323515       0.00      0.00      0.00         1
     1350000       0.00      0.00      0.00         4
     1350500       0.00      0.00      0.00         1
     1363000       0.00      0.00      0.00         1
     1366682       0.00      0.00      0.00         1
     1368157       0.00      0.00      0.00         1
     1376848       0.00      0.00      0.00         1
     1377000       0.00      0.00      0.00         1
     1380700       0.00      0.00      0.00         1
     1399012       0.00      0.00      0.00         1
     1400000       0.00      0.00      0.00        19
     1400450       0.00      0.00      0.00         1
     1410000       0.00      0.00      0.00         2
     1410901       0.00      0.00      0.00         1
     1420000       0.00      0.00      0.00         1
     1424811       0.00      0.00      0.00         1
     1450000       0.00      0.00      0.00         4
     1460000       0.00      0.00      0.00         1
     1465000       0.00      0.00      0.00         1
     1465654       0.00      0.00      0.00         1
     1469623       0.00      0.00      0.00         1
     1472367       0.00      0.00      0.00         1
     1474090       0.00      0.00      0.00         1
     1475000       0.00      0.00      0.00         1
     1485510       0.00      0.00      0.00         1
     1491250       0.00      0.00      0.00         1
     1499998       0.00      0.00      0.00         1
     1499999       0.00      0.00      0.00         1
     1500000       0.00      0.00      0.00        95
     1500001       0.00      0.00      0.00         1
     1503411       0.00      0.00      0.00         1
     1507645       0.00      0.00      0.00         1
     1510000       0.00      0.00      0.00         1
     1520534       0.00      0.00      0.00         1
     1521346       0.00      0.00      0.00         1
     1522481       0.00      0.00      0.00         1
     1525000       0.00      0.00      0.00         1
     1530000       0.00      0.00      0.00         1
     1530625       0.00      0.00      0.00         1
     1532000       0.00      0.00      0.00         1
     1537591       0.00      0.00      0.00         1
     1543920       0.00      0.00      0.00         3
     1544067       0.00      0.00      0.00         1
     1550000       0.00      0.00      0.00         3
     1553727       0.00      0.00      0.00         1
     1554158       0.00      0.00      0.00         1
     1557478       0.00      0.00      0.00         1
     1564311       0.00      0.00      0.00         1
     1568000       0.00      0.00      0.00         1
     1600000       0.00      0.00      0.00        21
     1600500       0.00      0.00      0.00         1
     1602870       0.00      0.00      0.00         1
     1611500       0.00      0.00      0.00         1
     1625213       0.00      0.00      0.00         1
     1628000       0.00      0.00      0.00         1
     1630000       0.00      0.00      0.00         2
     1630235       0.00      0.00      0.00         1
     1636120       0.00      0.00      0.00         1
     1639987       0.00      0.00      0.00         1
     1640690       0.00      0.00      0.00         1
     1647000       0.00      0.00      0.00         1
     1648351       0.00      0.00      0.00         1
     1650000       0.00      0.00      0.00         6
     1666776       0.00      0.00      0.00         1
     1680511       0.00      0.00      0.00         1
     1700000       0.00      0.00      0.00        22
     1720000       0.00      0.00      0.00         1
     1725000       0.00      0.00      0.00         1
     1738250       0.00      0.00      0.00         1
     1740000       0.00      0.00      0.00         1
     1742701       0.00      0.00      0.00         1
     1750000       0.00      0.00      0.00        10
     1750006       0.00      0.00      0.00         1
     1751957       0.00      0.00      0.00         1
     1755000       0.00      0.00      0.00         1
     1765000       0.00      0.00      0.00         2
     1775000       0.00      0.00      0.00         1
     1795958       0.00      0.00      0.00         1
     1800000       0.00      0.00      0.00        20
     1800002       0.00      0.00      0.00         1
     1810380       0.00      0.00      0.00         1
     1840000       0.00      0.00      0.00         2
     1850000       0.00      0.00      0.00         5
     1867970       0.00      0.00      0.00         1
     1875000       0.00      0.00      0.00         3
     1880000       0.00      0.00      0.00         1
     1884000       0.00      0.00      0.00         1
     1885000       0.00      0.00      0.00         1
     1894000       0.00      0.00      0.00         1
     1894064       0.00      0.00      0.00         2
     1895000       0.00      0.00      0.00         1
     1896024       0.00      0.00      0.00         1
     1899177       0.00      0.00      0.00         1
     1900000       0.00      0.00      0.00         8
     1911375       0.00      0.00      0.00         1
     1913654       0.00      0.00      0.00         1
     1929900       0.00      0.00      0.00         1
     1934000       0.00      0.00      0.00         1
     1938292       0.00      0.00      0.00         1
     1940000       0.00      0.00      0.00         1
     1947189       0.00      0.00      0.00         1
     1950000       0.00      0.00      0.00         3
     1968000       0.00      0.00      0.00         1
     1981872       0.00      0.00      0.00         1
     1986990       0.00      0.00      0.00         1
     2000000       0.00      0.00      0.00       109
     2000290       0.00      0.00      0.00         1
     2013334       0.00      0.00      0.00         1
     2018000       0.00      0.00      0.00         1
     2025000       0.00      0.00      0.00         2
     2026468       0.00      0.00      0.00         1
     2044050       0.00      0.00      0.00         1
     2050000       0.00      0.00      0.00         3
     2071450       0.00      0.00      0.00         1
     2075000       0.00      0.00      0.00         1
     2079999       0.00      0.00      0.00         1
     2099700       0.00      0.00      0.00         1
     2100000       0.00      0.00      0.00        14
     2100505       0.00      0.00      0.00         1
     2104119       0.00      0.00      0.00         1
     2114831       0.00      0.00      0.00         1
     2115375       0.00      0.00      0.00         1
     2120000       0.00      0.00      0.00         2
     2123820       0.00      0.00      0.00         1
     2125000       0.00      0.00      0.00         1
     2150000       0.00      0.00      0.00         1
     2162250       0.00      0.00      0.00         1
     2168237       0.00      0.00      0.00         1
     2175000       0.00      0.00      0.00         1
     2180175       0.00      0.00      0.00         1
     2190000       0.00      0.00      0.00         1
     2200000       0.00      0.00      0.00        12
     2210000       0.00      0.00      0.00         2
     2219302       0.00      0.00      0.00         1
     2225623       0.00      0.00      0.00         1
     2226000       0.00      0.00      0.00         1
     2243000       0.00      0.00      0.00         1
     2250000       0.00      0.00      0.00         6
     2250180       0.00      0.00      0.00         1
     2256555       0.00      0.00      0.00         1
     2275000       0.00      0.00      0.00         1
     2284354       0.00      0.00      0.00         1
     2300000       0.00      0.00      0.00        13
     2310008       0.00      0.00      0.00         1
     2333534       0.00      0.00      0.00         1
     2350000       0.00      0.00      0.00         1
     2360000       0.00      0.00      0.00         1
     2370000       0.00      0.00      0.00         1
     2400000       0.00      0.00      0.00         5
     2416756       0.00      0.00      0.00         1
     2420000       0.00      0.00      0.00         1
     2425000       0.00      0.00      0.00         1
     2449999       0.00      0.00      0.00         1
     2450000       0.00      0.00      0.00         7
     2489905       0.00      0.00      0.00         1
     2500000       0.00      0.00      0.00        32
     2520000       0.00      0.00      0.00         2
     2525000       0.00      0.00      0.00         1
     2528000       0.00      0.00      0.00         1
     2531304       0.00      0.00      0.00         1
     2542288       0.00      0.00      0.00         1
     2550000       0.00      0.00      0.00         2
     2553484       0.00      0.00      0.00         1
     2570000       0.00      0.00      0.00         1
     2600000       0.00      0.00      0.00         5
     2605104       0.00      0.00      0.00         1
     2610000       0.00      0.00      0.00         1
     2612879       0.00      0.00      0.00         1
     2620000       0.00      0.00      0.00         2
     2621250       0.00      0.00      0.00         1
     2628000       0.00      0.00      0.00         1
     2638000       0.00      0.00      0.00         1
     2640000       0.00      0.00      0.00         1
     2650000       0.00      0.00      0.00         3
     2670000       0.00      0.00      0.00         1
     2675790       0.00      0.00      0.00         1
     2698918       0.00      0.00      0.00         1
     2700000       0.00      0.00      0.00         7
     2705000       0.00      0.00      0.00         1
     2715000       0.00      0.00      0.00         1
     2720000       0.00      0.00      0.00         1
     2748531       0.00      0.00      0.00         1
     2750000       0.00      0.00      0.00         3
     2785830       0.00      0.00      0.00         1
     2800000       0.00      0.00      0.00         7
     2811110       0.00      0.00      0.00         1
     2822067       0.00      0.00      0.00         1
     2825000       0.00      0.00      0.00         1
     2878032       0.00      0.00      0.00         1
     2900000       0.00      0.00      0.00         2
     2950000       0.00      0.00      0.00         1
     2985250       0.00      0.00      0.00         1
     2999999       0.00      0.00      0.00         1
     3000000       0.00      0.00      0.00        46
     3020000       0.00      0.00      0.00         3
     3030502       0.00      0.00      0.00         1
     3047000       0.00      0.00      0.00         1
     3050000       0.00      0.00      0.00         1
     3100000       0.00      0.00      0.00         3
     3109282       0.00      0.00      0.00         1
     3150000       0.00      0.00      0.00         1
     3161435       0.00      0.00      0.00         1
     3170455       0.00      0.00      0.00         1
     3200000       0.00      0.00      0.00         6
     3210000       0.00      0.00      0.00         1
     3211700       0.00      0.00      0.00         1
     3216500       0.00      0.00      0.00         1
     3222000       0.00      0.00      0.00         1
     3250000       0.00      0.00      0.00         7
     3262320       0.00      0.00      0.00         1
     3272251       0.00      0.00      0.00         1
     3295000       0.00      0.00      0.00         1
     3300000       0.00      0.00      0.00         4
     3320000       0.00      0.00      0.00         1
     3360000       0.00      0.00      0.00         1
     3375000       0.00      0.00      0.00         1
     3400000       0.00      0.00      0.00         6
     3403750       0.00      0.00      0.00         1
     3427500       0.00      0.00      0.00         1
     3450000       0.00      0.00      0.00         1
     3453000       0.00      0.00      0.00         1
     3473820       0.00      0.00      0.00         1
     3475109       0.00      0.00      0.00         1
     3485000       0.00      0.00      0.00         1
     3500000       0.00      0.00      0.00        12
     3528000       0.00      0.00      0.00         1
     3536162       0.00      0.00      0.00         1
     3550000       0.00      0.00      0.00         1
     3565000       0.00      0.00      0.00         1
     3568274       0.00      0.00      0.00         1
     3600000       0.00      0.00      0.00         2
     3680000       0.00      0.00      0.00         1
     3700000       0.00      0.00      0.00         2
     3740000       0.00      0.00      0.00         1
     3750000       0.00      0.00      0.00         2
     3770000       0.00      0.00      0.00         1
     3800800       0.00      0.00      0.00         1
     3804991       0.00      0.00      0.00         1
     3830000       0.00      0.00      0.00         1
     3837860       0.00      0.00      0.00         1
     3850000       0.00      0.00      0.00         1
     3874996       0.00      0.00      0.00         1
     3889691       0.00      0.00      0.00         1
     3891000       0.00      0.00      0.00         1
     3900000       0.00      0.00      0.00         1
     3980000       0.00      0.00      0.00         1
     3987693       0.00      0.00      0.00         1
     3999997       0.00      0.00      0.00         1
     3999999       0.00      0.00      0.00         1
     4000000       0.00      0.00      0.00        18
     4014726       0.00      0.00      0.00         1
     4050000       0.00      0.00      0.00         1
     4099999       0.00      0.00      0.00         1
     4100000       0.00      0.00      0.00         1
     4102097       0.00      0.00      0.00         1
     4115000       0.00      0.00      0.00         1
     4169942       0.00      0.00      0.00         1
     4300000       0.00      0.00      0.00         1
     4410000       0.00      0.00      0.00         1
     4410717       0.00      0.00      0.00         1
     4499568       0.00      0.00      0.00         1
     4500000       0.00      0.00      0.00        10
     4600000       0.00      0.00      0.00         1
     4618000       0.00      0.00      0.00         1
     4645311       0.00      0.00      0.00         1
     4700000       0.00      0.00      0.00         1
     4700610       0.00      0.00      0.00         1
     4710960       0.00      0.00      0.00         1
     4875000       0.00      0.00      0.00         1
     4943554       0.00      0.00      0.00         1
     5000000       0.00      0.00      0.00         7
     5010000       0.00      0.00      0.00         1
     5150000       0.00      0.00      0.00         1
     5186000       0.00      0.00      0.00         1
     5250000       0.00      0.00      0.00         1
     5303380       0.00      0.00      0.00         1
     5399988       0.00      0.00      0.00         1
     5400000       0.00      0.00      0.00         1
     5492627       0.00      0.00      0.00         1
     5500000       0.00      0.00      0.00         3
     5500003       0.00      0.00      0.00         1
     5600000       0.00      0.00      0.00         3
     5750000       0.00      0.00      0.00         1
     5800000       0.00      0.00      0.00         1
     5900008       0.00      0.00      0.00         1
     6000000       0.00      0.00      0.00         2
     6000063       0.00      0.00      0.00         1
     6280000       0.00      0.00      0.00         1
     6300000       0.00      0.00      0.00         1
     6429018       0.00      0.00      0.00         1
     6448062       0.00      0.00      0.00         1
     6454000       0.00      0.00      0.00         1
     6500000       0.00      0.00      0.00         1
     6600000       0.00      0.00      0.00         1
     6706000       0.00      0.00      0.00         1
     7000000       0.00      0.00      0.00         1
     7500000       0.00      0.00      0.00         2
     7800000       0.00      0.00      0.00         1
     8000000       0.00      0.00      0.00         3
     8065397       0.00      0.00      0.00         1
     8500000       0.00      0.00      0.00         1
     8700000       0.00      0.00      0.00         1
     9080000       0.00      0.00      0.00         1
    10000000       0.00      0.00      0.00         1
    21000000       0.00      0.00      0.00         1
    64000000       0.00      0.00      0.00         1

    accuracy                           0.72     16315
   macro avg       0.00      0.00      0.00     16315
weighted avg       0.52      0.72      0.61     16315

[[11790     0     0 ...     0     0     0]
 [    1     0     0 ...     0     0     0]
 [    1     0     0 ...     0     0     0]
 ...
 [    1     0     0 ...     0     0     0]
 [    1     0     0 ...     0     0     0]
 [    1     0     0 ...     0     0     0]]
0.7226478700582286
_train
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
_train
print(f'Model Accuracy: {tree.score(x_train, y_train)}')
Model Accuracy: 0.7243222027655335
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(x_train, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
# Actual class predictions
rf_predictions = model.predict(x_test)
# Probabilities for each class
rf_probs = model.predict_proba(x_test)
model
model.score(x_test, y_test)
0.7190315660435183
​
