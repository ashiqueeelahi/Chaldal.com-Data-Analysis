import pandas as pd
fish = pd.read_csv('../input/chaldalcom-groceries-dataset/chaldal-fish.csv');
fruits = pd.read_csv('../input/chaldalcom-groceries-dataset/chaldal-fruits.csv');
meat = pd.read_csv('../input/chaldalcom-groceries-dataset/chaldal-meat.csv');
veg = pd.read_csv('../input/chaldalcom-groceries-dataset/chaldal-vegetable.csv')

import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
import pickle;

Fish

Let us look at our data

fish.head()




People are buying fish according to their need as well as ability. We can visualize, in what qantities people are buying most and least

plt.figure(figsize = (20,10))
plt.figure(figsize = (16,9))
Quantity = fish['Quantity'].value_counts().reset_index()
plt.pie(Quantity['Quantity'], labels = Quantity['index'],autopct='%1.1f%%', startangle=45)
plt.title("Percentage of different amount of fish bought by people")
plt.show()



Here we can see, most fish are bought in 500 gms of quantity. If we have to pick the second best value in terms of quantity, it would be 1kg benchmark. So, we can take this range as standard. (500gm - 1kg). Around 70% people are buying fish of that quantity.

As for the least values, they lie in the range of just below that 500 gm benchmark and 1 kg benchmark.

In our data file, price value is not really for visualization. We have to preprocess it first

fish['Price'] = fish['Price'].replace({'৳':''}, regex = True)

fish['Price'] = fish['Price'].replace({'1,': '1'}, regex = True)

fish['Price'] = fish['Price'].astype('int32')

fish.head(16)



Now its perfect

plt.figure(figsize = (16,9))
b = [ 50, 100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500];
sns.distplot( fish['Price'],  bins = b, color = 'R',)
plt.xticks(b, fontsize = 13)
plt.title("Price Range for Different Kind of Fishes", fontsize = 25)
plt.xlabel("Price Range", fontsize = 18)
plt.ylabel("Count", fontsize = 18)

plt.show()



Nice. We have our maximum values in (200taka - 500taka) range. So, people are mostly spending that amount of money to buy fish. As for the second best is concerned, it lies in (50taka-200taka) and (600taka - 700taka) range. Which is also located in the circumference of our standard (200taka -500taka) range. And, one more interesting insight. Very less amount are spend to buy fish at (1000taka+) range.

plt.figure(figsize = (16,9))
sns.kdeplot(data= fish['Price'], shade = True, color = 'G')
plt.title("Price Range for Different Kind of Fishes")
plt.xlabel("Price Range")
plt.ylabel("Count")

plt.show()



yeah. This graph also tells us the same story. The standard range of money spent for fish is (200taka - 500taka)
Fruits

fruits.head()

fruits['Category'].unique()

array(['Fruit'], dtype=object)

fruits['Quantity'].unique()



Let us visualize, in what quantity people are buying fruits

plt.figure(figsize = (20,10))
plt.figure(figsize = (16,9));
textprops = {'fontsize': 15}
Quantity = fruits['Quantity'].value_counts().reset_index()
plt.pie(Quantity['Quantity'], labels = Quantity['index'],autopct='%1.1f%%', startangle=45, textprops = textprops)
plt.title("Percentage of different amount of fruits bought by people", fontsize = 25)
plt.show()



So, people are mostly buying 1kg fruit. But, the second and third best values are surprising. Unlike fish, People are not always buying fruits in kgs

So, Not every fruit is being sold kgs. Some are also sold in dozens. Moreover, some fruits are sold per piece. Lets figure out which are those fruits

Fruits_sold_per_piece = fruits.loc[fruits['Quantity'] == 'each']

Fruits_sold_per_piece['Product']



Pineapple, Coconut and Pomelo fruit are sold per piece. Pomelo Fruit is called Jambura in Bengali

Fruits_sold_per_12_piece = fruits.loc[fruits['Quantity'] == '12 pcs'];
Fruits_sold_per_12_piece.head()



Nice. As expected I would say. Banana is sold in dozens.

Just like the price data from fish, we have to preprocess fruit price data as well for visualization.

fruits['Price'] = fruits['Price'].replace({'৳':''}, regex = True)
fruits['Price'] = fruits['Price'].astype('int32')

Now we can plot them

plt.figure(figsize = (16,9))
b = [25, 50, 100,200,300,400,450,500,600];
sns.distplot( fruits['Price'],  bins = b, color = 'G',)
plt.xticks(b,fontsize = 15)
plt.title("Price Range for Different Kind of Fruits", fontsize = 25)
plt.xlabel("Price Range", fontsize = 18)
plt.ylabel("Count", fontsize = 18)

plt.show()



So, most fruit are bought in the price range of (50taka -100taka). We can take it as a standard value. Because it seems to be a clear winner in terms of percentage. As for the second best value is concerned, it lies between (100taka - 200taka)
Meat

meat.head()

meat['Category'].unique()



Lets explore in what quantity people are buying meat

plt.figure(figsize = (20,10))
plt.figure(figsize = (16,9));
textprops = {'fontsize': 15}
Quantity = meat['Quantity'].value_counts().reset_index()
plt.pie(Quantity['Quantity'], labels = Quantity['index'],autopct='%1.1f%%', startangle=45, textprops = textprops)
plt.title("Percentage of different amount of meat bought by people", fontsize = 25)
plt.show()




The standard value is 1kg. Very few people are buying meat in an amount other than that

As usual, lets preprocess the price data for plotting and visualization

meat['Price'] = meat['Price'].replace({'৳':''}, regex = True)
meat['Price'] = meat['Price'].astype('int32')

plt.figure(figsize = (20,10))
b = [100,200,300,400,500,600,700,800,900,1000,1100];
sns.distplot( meat['Price'],  bins = b, color = 'B',)
plt.xticks(b, fontsize = 15)
plt.title("Price Range for Different Kind of Meat", fontsize = 25)
plt.xlabel("Price Range", fontsize = 18)
plt.ylabel("Count", fontsize = 18)

plt.show()


Vegetable

veg.head()

plt.figure(figsize = (20,10))
plt.figure(figsize = (16,9));
explode = [0, 0.2,0,0.3,0,0.3,0,0, 0,0];
Quantity = veg['Quantity'].value_counts().reset_index()
plt.pie(Quantity['Quantity'], labels = Quantity['index'],explode = explode, autopct='%1.1f%%', startangle=45 )
plt.title("Percentage of different amount of vegetable bought by people", fontsize = 25)
plt.show()



Most are bought in 500gms of amount. But a good percentage is also bought in 1kg amount. We have similarities here as fruit. Some people are buying vegetables in dozens, pieces and 4 pieces. Lets find out which are those

veg_sold_per_piece = veg.loc[veg['Quantity'] == 'each'];
veg_sold_per_piece['Product']



So, people are buying Cabbage, Gourd, Aloe Vera and different type of Pumkins in pieces.

veg_sold_per_bundle = veg.loc[veg['Quantity'] == '1 bundle'];
veg_sold_per_bundle['Product']



Data shak, Pennywort leaves and different kind of spinach are bought in bundles.

veg_sold_per_4_piece = veg.loc[veg['Quantity'] == '4 pcs'];
veg_sold_per_4_piece['Product']



Different kind of lemons and green bananas(bananas that are not ripe yet) are bought in 4 pieces. In Bangladesh, 4 pieces are called hali. So, going to shop, one would say "Give me one hali of lemon", which means, 'give me 4 pieces of lemon'

Preprocessing the price data

veg['Price'] = veg['Price'].replace({'৳':''}, regex = True)
veg['Price'] = veg['Price'].astype('int32')

plt.figure(figsize = (20,10))
b = [0,10,20,30,40,50,60,70,80,90,100,110, 120,130,140,150,160,170,180,190,200,210,220,230,240,250];
sns.distplot( veg['Price'],  bins = b, color = 'B',)
plt.xticks(b, fontsize = 15)
plt.title("Price Range for Different Kind of Vegetable", fontsize = 25)
plt.xlabel("Price Range", fontsize = 18)
plt.ylabel("Count", fontsize = 18)

plt.show()

