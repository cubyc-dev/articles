# Predicting Premier League xG using Cubyc's Anterior engine.
###### We built a model to predict the winner (in xG) of any Premier League match, using Python, XGBoost, and Cubyc’s Anterior engine.

---

<img src="stadium.png" alt="We built a model that can accurately predict the xG winner in 80% of games every match day, using Python, XGBoost, and Cubyc's Anterior engine." width="750"/>

##### What are we building today? 
We are building a model to predict the expected goals (xG) winner in a Premier League game. xG has been accepted by the data science and footballing world alike for its ability to provide insight into which team played better football on any given day. This provides the continuous and chaotic game we all love with some structure and interpretative power.

Aggregated, this information can be used to assess a team's performance over a season, as famed xG-evangelist [@xGPhilosophy](https://twitter.com/xGPhilosophy) does in his Expected Points vs. Realized Points table.

![X users arm themselves with xG tables in ongoing debates against their bitter rivals. Source: @xGPhilosophy](xgtable.jpeg)

> Aggregating this information is called backtesting - measuring the impact of a change in your primary KPI.

We've crunched the numbers to build a boosted tree model that predicts who'll win in xG on every match day. 

Here's how we did it:


##### Overview
1. Gathering the data 💽
2. Setting up Anterior 🔌
3. v0 Model 🧑‍💻
4. Moving forward 🏃‍♀️

---

## (1/4) Gathering the data 💽
For this model, we used [Football-Data](https://medium.com/r/?url=https%3A%2F%2Fwww.football-data.co.uk%2Fenglandm.php) historical PL results and [Transfermarkt](https://medium.com/r/?url=https%3A%2F%2Fwww.transfermarkt.us%2Fpremier-league%2Fstartseite%2Fwettbewerb%2FGB1) team attributes for the past two seasons. 

#### We want a data frame that looks like this:

| Date      | Home Team | Away Team | 3x Odds H,D,A | Team Attrbutes |
| ----------- | ----------- | ----------- | ----------- | ----------- |
|Aug 5, 2022 | Crystal Palace | Arsenal | 4.39, 3.59, 1.88 | Value(￡), Avg. Age, Num. of Foreigners  |

The basic premise is that we'll use static inputs as baseline predictors. This includes **Market Value (￡), Age and # of foreign players per team.**

We'll use dynamic inputs such as the betting odds and the month the match was played for statistical stability as the season progresses. This is because Palace vs. Arsenal on match day 1 may not have the same implications as on match day 38. Teams fight for relegation, European competition berths, and the title towards the last match days (38 total).

##### ⚠️ You'll need to encode your columns accordingly for XGBoost like this.
Note that one-hot encoding is necessary for categorical features. Your data frame will need to include boolean values for every team both home and away:

| Date | h_team_Palace | a_team_Arsenal | h_odds | d_odds | a_odds | h_val| a_val |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|2022-08-05 | TRUE | TRUE | 4.39 | 3.59 | 1.88 | 323.05	| 1000 |

---

## (2/4) Setting up Anterior🔌

##### What's Cubyc and how is it useful?  
> Cubyc's standout feature is its back tester - the ability to ask your data:
_"What would have happened had I implemented this strategy in the past?"_

We call this the Anterior engine.

```bash
pip install anterior
```

##### Import the main dependencies
```python
# import the backtester
from anterior import BackTester,  OracleDataFrame
# import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
```
---

##  (3/4) v0 Model 🧑‍💻
Now, build your model inside a function; this will allow us to retrain an infinite amount of times based on dates passed onto Anterior's `OracleDataFrame`.

```python
# load your data into an Anterior OracleDataFrame
odf = OracleDataFrame.from_csv("pl_data.csv")

# write your boilerplate model
def retrain():
    df = odf.reset_index().drop(["Date", "result"], axis=1)
    X = df
    y = odf["result"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # TRAIN MODEL
    # start XGBoost classifier
    model = xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss")
    model.fit(X_train, y_train)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_test, y_pred)

    metrics.append((y_pred, accuracy))
```

Ok, that's all the heavy lifting. Running the back tester is as easy as :
```python
bt = BackTester()
metrics = []

# train model seasonally and up to the 2022/2023 season
bt.on("2023-05-24").do(retrain)
bt.run(start=("2022-08-15", end="2023-05-24")

# our model!
preds = [pred for pred, _ in metrics] # prediction array
accs = [acc for _,acc in metrics] # model performance 52.5%
```

>Typically you'd need to create a sort of for-loop or manually play with .iloc[], however, Anterior's OracleDataFrames handle all of that in the background. Under the hood, Anterior is dynamically exposing data to the model, up to each datetime passed on bt.on(datetime) .
---

#### What just happened?

You just built production-level ML architecture in less than 15 lines of code! By passing an `OracleDataFrame` into `bt.every.do`, Anterior pegged itself to our datetime column. 

**Why?** _Not all Premier League match days are the same.
With Anterior's Oracle data structures, you can dynamically access your DataFrame and Series objects on-the-fly during backtests._

Continually backtesting and evaluating model performance allows us to slice the season into chunks, be it entire seasons, season start, middle, and run-in.

<img src="players_snow.gif" width=380/>

---
### Moving forward 🏃‍♀️
Let's go back to our code and see how Anterior keeps you in a flexible workflow. Here's a few examples:
>Train a model every week since the start of 2022/23 season

```python
# reset
bt, metrics = BackTester(), []

# train a model every 7 days
bt.every(days=7).do(retrain)
bt.run(start="2022-08-15", end="2024-04-01")

# our model!
preds = [pred for pred, _ in metrics] # prediction array
accs = [acc for _,acc in metrics] # model performance 58.8%
```

>Train a model every month since the start of 2022/23 season

```python
# reset once more
bt, metrics = BackTester(), []

# train a model every month
bt.every(months=1).do(retrain)
bt.run(start="2022-08-15", end="2024-04-01")

# our model!
preds = [pred for pred, _ in metrics] # prediction array
accs = [acc for _,acc in metrics] # model performance 62.5%
```

>Train a model based on an array

```python
# important periods through the 22/23 & 23/24 seasons
season = [("2022-08-15", "2023-12-01"), ("2024-01-01", "2024-04-02")]

bt = BackTester()
metrics = []

# train model seasonally predict up to April 2024
bt.on(season[1][2]).do(retrain)
bt.run(start=(season[0][0], end=season[1][1])

# our model! ... you get the gist.
preds = [pred for pred, _ in metrics] # prediction array
accs = [acc for _,acc in metrics] # model performance 52.5%
````

You can now maneuver in time throughout the season given the requirements from your data team. The Anterior engine allows you to maintain a flexible architecture centered around human decisions - as opposed to one constrained by data parsing and adhoc for-loops.



---
Cubyc is an open-source python framework for data scientists and engineers. If you found this article helpful, we'd love to hear your feedback, feel free to get in contact with our team below: 

For more on Cubyc and the Anterior © engine

* [Docs](https://docs.cubyc.com/anterior) for reference
* [Join our Discord!](https://discord.gg/e7zU8xcZ2E)
* [Follow us on LinkedIn](https://www.linkedin.com/company/cubyc/) if you're planning to build on top of Cubyc!
* Cubyc.com/anterior to learn more about Cubyc