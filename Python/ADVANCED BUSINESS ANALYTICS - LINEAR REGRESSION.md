# ADVANCED BUSINESS ANALYTICS: LINEAR REGRESSION

In this assignment we are continuing to work with customer reward programs. An analyst performed some preliminary data preprocessing on the raw data and shared the data with you in the file crp_clean.xlsx. Note that some additional columns are created and some data columns are scaled. 

In this exercise, you will complete a predictive modeling task where the target variable is continuous based on the data in the shared file. 

First remove all rows where either the Reward or NumStores column takes the value 0. Also remove all rows where the rewards do not expire (ExpirationMonth=999). Consider linear regression models with ExpirationMonth column as the target variable.


```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn.linear_model 
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```


```python
data = pd.read_csv('crp_clean.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Retailer</th>
      <th>Salerank</th>
      <th>X2013USSales</th>
      <th>X2013WorldSales</th>
      <th>ProfitMargin</th>
      <th>NumStores</th>
      <th>Industry</th>
      <th>Reward</th>
      <th>ProgramName</th>
      <th>RewardType</th>
      <th>RewardStructure</th>
      <th>RewardSize</th>
      <th>ExpirationMonth</th>
      <th>IndustryType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A&amp;P</td>
      <td>74.0</td>
      <td>5.831</td>
      <td>5.831</td>
      <td>48.85</td>
      <td>0.277</td>
      <td>Discount, Variety Stores</td>
      <td>0.0</td>
      <td>No rewards program</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>Discount</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albertsons</td>
      <td>21.0</td>
      <td>19.452</td>
      <td>19.452</td>
      <td>69.02</td>
      <td>1.024</td>
      <td>Grocery Stores</td>
      <td>0.0</td>
      <td>No rewards program</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>Grocery</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aldi</td>
      <td>38.0</td>
      <td>10.898</td>
      <td>10.650</td>
      <td>69.41</td>
      <td>1.328</td>
      <td>Grocery Stores</td>
      <td>0.0</td>
      <td>No rewards program</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>Grocery</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alimentation Couche Tard (Circle K)</td>
      <td>82.0</td>
      <td>4.755</td>
      <td>8.551</td>
      <td>68.03</td>
      <td>3.826</td>
      <td>Grocery Stores</td>
      <td>0.0</td>
      <td>No rewards program</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>Grocery</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Apple Stores</td>
      <td>15.0</td>
      <td>26.648</td>
      <td>30.736</td>
      <td>11.07</td>
      <td>0.254</td>
      <td>ElectronicEquipment</td>
      <td>0.0</td>
      <td>No rewards program</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>Specialty</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = data[(data["Reward"] > 0) & (data["NumStores"] > 0) & (data["ExpirationMonth"] != "999")]
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Retailer</th>
      <th>Salerank</th>
      <th>X2013USSales</th>
      <th>X2013WorldSales</th>
      <th>ProfitMargin</th>
      <th>NumStores</th>
      <th>Industry</th>
      <th>Reward</th>
      <th>ProgramName</th>
      <th>RewardType</th>
      <th>RewardStructure</th>
      <th>RewardSize</th>
      <th>ExpirationMonth</th>
      <th>IndustryType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>7-Eleven</td>
      <td>35.0</td>
      <td>11.625</td>
      <td>11.504</td>
      <td>50.31</td>
      <td>7.974</td>
      <td>Discount, Variety Stores</td>
      <td>1.0</td>
      <td>7REWARDS</td>
      <td>store credit</td>
      <td>Buy 6 drinks  =  free drink</td>
      <td>16.67</td>
      <td>12</td>
      <td>Discount</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Advance Auto Parts</td>
      <td>66.0</td>
      <td>6.443</td>
      <td>6.485</td>
      <td>64.24</td>
      <td>4.023</td>
      <td>Auto Parts Stores</td>
      <td>1.0</td>
      <td>SpeedPerks</td>
      <td>store credit</td>
      <td>Spend $30/$100 = $5/$20 rewards for the next $...</td>
      <td>16.6</td>
      <td>2</td>
      <td>Specialty</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Ahold USA / Royal</td>
      <td>17.0</td>
      <td>26.118</td>
      <td>44.028</td>
      <td>40.87</td>
      <td>0.767</td>
      <td>Grocery Stores</td>
      <td>1.0</td>
      <td>Stop and Shop Rewards</td>
      <td>gas discount</td>
      <td>Spend $100 and earn 0.10 off per gallon for th...</td>
      <td>15</td>
      <td>1</td>
      <td>Grocery</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Ascena Retail Group (Ann Taylor)</td>
      <td>84.0</td>
      <td>4.665</td>
      <td>4.715</td>
      <td>93.45</td>
      <td>3.854</td>
      <td>Apparel Stores</td>
      <td>1.0</td>
      <td>PerfectRewards</td>
      <td>store credit</td>
      <td>Spend $400 earn 2,000 points = $20 PERFECT REW...</td>
      <td>5</td>
      <td>1</td>
      <td>Specialty</td>
    </tr>
    <tr>
      <th>51</th>
      <td>AutoZone</td>
      <td>56.0</td>
      <td>7.584</td>
      <td>15.190</td>
      <td>96.64</td>
      <td>4.802</td>
      <td>Auto Parts Stores</td>
      <td>1.0</td>
      <td>AutoZone Rewards</td>
      <td>store credit</td>
      <td>Spend $20+ 5 times earn 5 credits = $20 mercha...</td>
      <td>20</td>
      <td>3</td>
      <td>Specialty</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
data.reset_index(inplace = True, drop = True)
data['ExpirationMonth'] = data['ExpirationMonth'].astype(float)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Retailer</th>
      <th>Salerank</th>
      <th>X2013USSales</th>
      <th>X2013WorldSales</th>
      <th>ProfitMargin</th>
      <th>NumStores</th>
      <th>Industry</th>
      <th>Reward</th>
      <th>ProgramName</th>
      <th>RewardType</th>
      <th>RewardStructure</th>
      <th>RewardSize</th>
      <th>ExpirationMonth</th>
      <th>IndustryType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7-Eleven</td>
      <td>35.0</td>
      <td>11.625</td>
      <td>11.504</td>
      <td>50.31</td>
      <td>7.974</td>
      <td>Discount, Variety Stores</td>
      <td>1.0</td>
      <td>7REWARDS</td>
      <td>store credit</td>
      <td>Buy 6 drinks  =  free drink</td>
      <td>16.67</td>
      <td>12.0</td>
      <td>Discount</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Advance Auto Parts</td>
      <td>66.0</td>
      <td>6.443</td>
      <td>6.485</td>
      <td>64.24</td>
      <td>4.023</td>
      <td>Auto Parts Stores</td>
      <td>1.0</td>
      <td>SpeedPerks</td>
      <td>store credit</td>
      <td>Spend $30/$100 = $5/$20 rewards for the next $...</td>
      <td>16.6</td>
      <td>2.0</td>
      <td>Specialty</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ahold USA / Royal</td>
      <td>17.0</td>
      <td>26.118</td>
      <td>44.028</td>
      <td>40.87</td>
      <td>0.767</td>
      <td>Grocery Stores</td>
      <td>1.0</td>
      <td>Stop and Shop Rewards</td>
      <td>gas discount</td>
      <td>Spend $100 and earn 0.10 off per gallon for th...</td>
      <td>15</td>
      <td>1.0</td>
      <td>Grocery</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ascena Retail Group (Ann Taylor)</td>
      <td>84.0</td>
      <td>4.665</td>
      <td>4.715</td>
      <td>93.45</td>
      <td>3.854</td>
      <td>Apparel Stores</td>
      <td>1.0</td>
      <td>PerfectRewards</td>
      <td>store credit</td>
      <td>Spend $400 earn 2,000 points = $20 PERFECT REW...</td>
      <td>5</td>
      <td>1.0</td>
      <td>Specialty</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AutoZone</td>
      <td>56.0</td>
      <td>7.584</td>
      <td>15.190</td>
      <td>96.64</td>
      <td>4.802</td>
      <td>Auto Parts Stores</td>
      <td>1.0</td>
      <td>AutoZone Rewards</td>
      <td>store credit</td>
      <td>Spend $20+ 5 times earn 5 credits = $20 mercha...</td>
      <td>20</td>
      <td>3.0</td>
      <td>Specialty</td>
    </tr>
  </tbody>
</table>
</div>



## Question 1 

Find te model with one predictor variable and the highest R-squared. Consider the following set of predictor variables:
- Salerank
- X2013USSales
- X2013WorldSales
- RewardSize
- ProfitMargin


```python
features = ["Salerank", "X2013USSales", "X2013WorldSales", "RewardSize", "ProfitMargin"]
y = data["ExpirationMonth"].values.reshape(46, 1)

```


```python
lr = LR()
test_x = data['Salerank'].values.reshape(46, 1)
model = lr.fit(test_x, y)
```


```python
model.score(test_x, y)
```




    0.06215426746125208




```python
model.intercept_
```




    array([10.43886397])




```python
pred = model.predict(test_x)
```


```python
data.loc[38]['Retailer']
```




    'Subway'




```python
for feature in features:
    X = data[["NumStores", feature]].values.reshape(46, 2)
    model = lr.fit(X, y)
    predicted = model.predict(X)
    score = model.score(X, y)
    intercept = model.intercept_
    slope = model.coef_
    print (feature, ":", "R^2=", score, "intercept=", intercept, "slope=", slope)

```

    Salerank : R^2= 0.2706456819812242 intercept= [6.67687718] slope= [[ 0.83189299 -0.03491206]]
    X2013USSales : R^2= 0.2844758631770631 intercept= [5.85759906] slope= [[ 0.94050316 -0.06901207]]
    X2013WorldSales : R^2= 0.26716510415356076 intercept= [5.58882773] slope= [[ 0.90784555 -0.03828747]]
    RewardSize : R^2= 0.2843260587286368 intercept= [6.07700564] slope= [[ 0.90938412 -0.18817992]]
    ProfitMargin : R^2= 0.259580527324102 intercept= [5.82282779] slope= [[ 0.88417119 -0.02111462]]
    

Data transformation is a great way to improve model fit. Now consider the log transformation for the model identified in the previous question. You can choose to transform neither of them, one of them, or both of them. You should have four different models.

- Model 1: neither variable is transformed; this gives you the same model as in the previous question.
- Model 2: only the target variable is transformed
- Model 3: only the explanatory variable is transformed
- Model 4: both variables are transformed. Report the R-squared values of all four models.


```python
transformed_data = data.copy()
transformed_data = transformed_data[['NumStores', 'ExpirationMonth']]
transformed_data["LogY"] = np.log(transformed_data["ExpirationMonth"])
transformed_data["LogX"] = np.log(transformed_data["NumStores"])
lr = LR()

## MODEL 1

y_1 = transformed_data['ExpirationMonth'].values.reshape(46,1)
x_1 = transformed_data['NumStores'].values.reshape(46,1)
model_1 = lr.fit(x_1, y_1)
model_1.predict(x_1)
model_1.score(x_1, y_1)
```




    0.2537147187434302




```python
## MODEL 2

y_2 = transformed_data['LogY'].values.reshape(46,1)
x_2 = transformed_data['NumStores'].values.reshape(46,1)
model_2 = lr.fit(x_2, y_2)
model_2.predict(x_2)
model_2.score(x_2, y_2)
```




    0.07041983612024416




```python
## MODEL 3

y_3 = transformed_data['ExpirationMonth'].values.reshape(46,1)
x_3 = transformed_data['LogX'].values.reshape(46,1)
model_3 = lr.fit(x_3, y_3)
model_3.predict(x_3)
model_3.score(x_3, y_3)

```




    0.14469682613098556




```python
## MODEL 4

y_4 = transformed_data['LogY'].values.reshape(46,1)
x_4 = transformed_data['LogX'].values.reshape(46,1)
model_4 = lr.fit(x_4, y_4)
model_4.predict(x_4)
model_4.score(x_4, y_4)
```




    0.06526684762496393




```python
transformed_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NumStores</th>
      <th>ExpirationMonth</th>
      <th>LogY</th>
      <th>LogX</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.974</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>2.076186</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.023</td>
      <td>2.0</td>
      <td>0.693147</td>
      <td>1.392028</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.767</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>-0.265268</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.854</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.349112</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.802</td>
      <td>3.0</td>
      <td>1.098612</td>
      <td>1.569032</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.492</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>0.400118</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.684</td>
      <td>3.0</td>
      <td>1.098612</td>
      <td>-0.379797</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.201</td>
      <td>6.0</td>
      <td>1.791759</td>
      <td>-1.604450</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.288</td>
      <td>3.0</td>
      <td>1.098612</td>
      <td>0.253091</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.309</td>
      <td>4.0</td>
      <td>1.386294</td>
      <td>0.269263</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7.621</td>
      <td>1.5</td>
      <td>0.405465</td>
      <td>2.030908</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.644</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>-0.440057</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.296</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>-1.217396</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4.272</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>1.452082</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2.432</td>
      <td>24.0</td>
      <td>3.178054</td>
      <td>0.888714</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.420</td>
      <td>2.0</td>
      <td>0.693147</td>
      <td>-0.867501</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.216</td>
      <td>1.5</td>
      <td>0.405465</td>
      <td>-1.532477</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.311</td>
      <td>3.0</td>
      <td>1.098612</td>
      <td>-1.167962</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.235</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>-1.448170</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.203</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>-1.594549</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.087</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>0.083422</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.158</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>0.146694</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3.519</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>1.258177</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2.648</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>0.973805</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.202</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>-1.599488</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.085</td>
      <td>6.0</td>
      <td>1.791759</td>
      <td>-2.465104</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.260</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>-1.347074</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.070</td>
      <td>2.0</td>
      <td>0.693147</td>
      <td>0.067659</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.823</td>
      <td>2.0</td>
      <td>0.693147</td>
      <td>-0.194799</td>
    </tr>
    <tr>
      <th>29</th>
      <td>4.166</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>1.426956</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.132</td>
      <td>2.0</td>
      <td>0.693147</td>
      <td>-2.024953</td>
    </tr>
    <tr>
      <th>31</th>
      <td>4.587</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>1.523226</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1.335</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.288931</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.226</td>
      <td>3.0</td>
      <td>1.098612</td>
      <td>-1.487220</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1.905</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>0.644482</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.471</td>
      <td>6.0</td>
      <td>1.791759</td>
      <td>0.385942</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1.515</td>
      <td>3.0</td>
      <td>1.098612</td>
      <td>0.415415</td>
    </tr>
    <tr>
      <th>37</th>
      <td>11.513</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>2.443477</td>
    </tr>
    <tr>
      <th>38</th>
      <td>26.644</td>
      <td>36.0</td>
      <td>3.583519</td>
      <td>3.282564</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1.544</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.434376</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1.965</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.675492</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2.454</td>
      <td>24.0</td>
      <td>3.178054</td>
      <td>0.897719</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.868</td>
      <td>12.0</td>
      <td>2.484907</td>
      <td>-0.141564</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1.276</td>
      <td>3.0</td>
      <td>1.098612</td>
      <td>0.243730</td>
    </tr>
    <tr>
      <th>44</th>
      <td>4.494</td>
      <td>24.0</td>
      <td>3.178054</td>
      <td>1.502743</td>
    </tr>
    <tr>
      <th>45</th>
      <td>7.998</td>
      <td>6.0</td>
      <td>1.791759</td>
      <td>2.079192</td>
    </tr>
  </tbody>
</table>
</div>


