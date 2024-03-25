# Predicting Premier League xG using boosted trees & Cubyc's Historyc © engine.
---

<img src="stadium.png" alt="Midjourney - Arsenal FC players in a Premier League match in the style of Stanley Kubrick, hyperrealistic, 4k, wide angle lens, cinematic lighting." width="750"/>

#### Overview
##### We built a model that can accurately predict the xG winner in 80% of games every matchday, using Python, XGBoost, and the Historyc © engine. 

Here are the steps: 
1. Gathering the data 💽
2. Plugging Cubyc 🔌
3. v0 Model 🧑‍💻
4. v1..n Models 📈
5. Moving forward 🏃‍♀️

Skip to data, model, and script downloads below _here_ .

---

#### (1/5) Gathering the data 💽
For this model we are using Football-Data historical PL results and Transfermarkt team attributes for the past two seasons. We want a dataframe that looks like this: 

| Date      | Home Team | Away Team | 3x Odds H,D,A | Team Attrbutes |
| ----------- | ----------- | ----------- | ----------- | ----------- |
|Aug 5, 2022 | Crystal Palace | Arsenal | 4.39, 3.59, 1.88 | Value(￡), Avg. Age, Num. of Foreigners  |

The basic premise is, we'll use static inputs as baseline predictors. This includes Market Value (￡), Avg. Age, and # of foreign players per team.

We'll use dynamic inputs such as the betting odds and the month the match was played in for statistical stability as the season progresses. The reason for this is that Palace vs Arsenal on the first day of the season may not have the same implications as on the last matchday. Teams fight for relegation, European competition berths, and the title itself in the last months.

##### ⚠️ You'll need to encode your columns accordingly for Xgboost like this. 
Note that one-hot encoding is necessary for categorical features.Your dataframe will actually include boolean values for every team both home and away.
| Date | h_team_Palace | a_team_Arsenal | h_odds | d_odds | a_odds | h_val| a_val |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|2022-08-05 | TRUE | TRUE | 4.39 | 3.59 | 1.88 | 323.05	| 1000 |

#### (2/5) Plugging Cubyc 🔌

##### What's Cubyc and how is it useful?  
Cubyc's standout feature is its backtester - the ability to ask your data: 
*"What would have happened had I implemented this strategy in the past?"*

We call this Historyc ©.

##### 

##### Install the Cubyc package and import historyc
```python
# install historyc
!git clone https://github.com/cubyc-team/historyc.git

# import the backtester
from historyc import BackTester,  OracleDataFrame
# other dependencies
import pandas as pd
import matplotlib.pyplot as plt
```

#####  Create a Cubyc DataFeed and train your model
```python
# load your data into a Historyc OracleDataFrame
odf = OracleDataFrame.from_csv("pl_data.csv")

# init the backtester
bt = BackTester()
# add your boilerplate model
def pl_season_retrain(data):
    metrics = []
    X = data.drop("result", axis=1)
    y = data["result"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # %%
    ### TRAIN MODEL
    # start XGBoost classifier
    model = xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    metrics.append(y_pred, accuracy)
```

Running the backtester is as easy as:
```python
@bt.do_every(months=1).do(retrain(odf))
```


#### What just happened?
You just built production-level ML architecture in less than 15 lines of code! By passing an `OracleDataFrame` into `@bt.do_every.do`, Historyc © pegged attached itself to our datetime column. Why? 

*Because not all Premier League matchdays are the same*. Continually backtesting and evaluating model performance is what allows us to slice the season into chunks, be it entire seasons, season start, middle, and run-in.

<img src="snow.png" width=380/>

### What's next? 
```python
# season start, mid-season, season run-in
season = [(2023-08-01, 2023,2023-11-01), 
(2023-12-01, 2024-03-01), 
(2024-03-02, 2024-05-31)]

@bt.do_every(season).do(retrain(odf))
```

You can now maneauver in time throughout the season given the requirements from your data team. The historyc engine allows you to maintain a flexible architecture centered around human decision - as opposed to one constrained by data parsing and adhoc for-loops.

---
