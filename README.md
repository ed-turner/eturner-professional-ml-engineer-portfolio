# Professional ML Engineer Portfolio

Welcome to my professional portfolio for all of my Machine Learning Engineering Work.

Throughout my years as a professional in the Data & AI space, I have learned how to properly deploy models given the different model architectures, automate the model training and capture the experiments, and handle specific feature engineering tasks.

The following sections will cover each of the different model architectures and how I would deploy those in different circumstances. 

## Feature Engineering

Each feature engineering task can be broken into two separate work streams: Pure Row-Altering ETL Operations (i.e. joins, filters) or Pure Columnar Operations (i.e. Polynormial features, Temporal shifts with missing values permitted).  We will discuss the each of these ways in more detail for clarification purposes.  Where these feature engineering task fit will be discussed in a later section. 

As a note that is important for each work stream, the engineered feature operation need to either be part of the Machine Learning Model artifact (i.e. part of pipeline object) or output the features into a versioned data output source.  

### Row-Altering ETL Operations

These Row-Altering ETL operations are generally employed at the beginnging of each Data Science Discovery work.  Row-Altering ETL operations include joining two tables, removing rows or adding rows.  Any other operations that does not include joins, filters or unions are considered Columnar based.  

The following are an example in the form of a SQL statement:

``` sql
SELECT * FROM EMP JOIN DEPT ON EMP.DEPTNO = DEPT.DEPTNO;
```

This same example can be shown as a set of pandas operations:

``` python

import pandas as pd

# this can be pd.read_parquet, pd.read_sql, etc
emp_df = pd.read_csv(...)

dept_df = pd.read_csv(...)

df = pd.merge(emp_df, dept_df, on="DEPTNO")


```

This same example can be shown as a set of spark operations (which is more flexible to either be deployed using scala or python):

``` python

# this spark session is created through the configurations you set
spark = ....

# this can be pd.read_parquet, pd.read_sql, etc
emp_sdf = spark.read.csv(...)

emp_sdf.createOrReplaceTempView("EMP")

dept_sdf = spark.read.csv(...)

dept_sdf.createOrReplaceTempView("DEPT")

sdf = spark.sql("SELECT * FROM EMP JOIN DEPT ON EMP.DEPTNO = DEPT.DEPTNO")

```

Each of the previous examples were of a join.  However, anything that is a filter or union will also fall under this.

### Pure Columnar Operations

Pure Columnar Operations are operations that does not change the row space, and only add columns and not mutate columns.  This can be incrementing a column by a constant value, standardizing the column, or adding a set of polynomial features.  

The following is an example of a pure columnar operation in SQL;

``` sql
SELECT *, age + 1 as age1 FROM EMP;
```

This same example can be shown as a set of pandas operations:

``` python

import pandas as pd

# this can be pd.read_parquet, pd.read_sql, etc
emp_df = pd.read_csv(...)

emp_df["age1"] = emp_df["age"] + 1

```

This same example can be shown as a set of spark operations (which is more flexible to either be deployed using scala or python):

``` python

import pyspark.sql.functions as F

# this spark session is created through the configurations you set
spark = ....

# this can be pd.read_parquet, pd.read_sql, etc
emp_sdf = spark.read.csv(...)

emp_sdf = emp_sdf.withColumn("age1", F.col("age") + F.lit(1))

```

One encouragement when performing purely columnar operations, especially when using sklearn or spark, is to use a Transformer object and add those into a pipeline object.  The rationale here is to enable proper A/B testing throughout the entire pipeline end to end.  

Here is an example of a sklearn standard feature engineering pipeline using Transformers:

``` python 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer

ct1 = ColumnTransformer(
  [ ("bucket", KBinsDiscretizer(), ["income", "age"]) ],
  )
  
ct2 = ColumnTransformer(
  [ ("onehot", OneHotEncoder(), ["deptno", "gender"]) ],
  remainder="passthrough"
  )

union = FeatureUnion([("numeric", ct1), ("ordinal", ct2)])

pipe = Pipeline([('sparse_features', union), ('standizer', StandardScaler())])

pipe.fit(df)

```

For incrementation feature engineering tasks, please consider using `sklearn.preprocessing.FunctionTransformer`.

One benefit, even if you deploying using a Spark pipeline, the feature engineering pipelines can be loaded into a `pyspark.sql.functions.pands_udf`.  
This will allow a scalable feature engineering pipeline, even if it wasn't developed using spark first.  


## Model Architectures

### Tensorflow Model

The `examples/tf-mle` folder will have the example of how I would deploy a tensorflow model.  Any subclass of `tf.keras.models.Model` and `tf.keras.models.Sequential` model can be deploy using the method described in this section.  The main deployment method is to use Kubernetes to deploy specific `tensorflow/serving` web services, which will specify a particular tensorflow model.

We will also cover how to automate hyperparameter tuning in a way that will save the parameters.

### Scikit-Learn Model

The `examples/mlflow-mle` folder will have the example of how I would deploy a scikit-learn model, or any model that was subclassing `sklearn.base.BaseEstimator`.  The main deployment method is to use Kubernetes to deploy a specific `mlflow` web service, which will specify a particular sklearn model.  

We will also cover how to automate the hyperparameter tuning using Bayesian Optimiztion.  

## Machine Learning Pipeline Flow

The Machine Learning Pipeline Flow is a very opininated space, where there are general disagreements for how to properly develop a Data Science methodology without committing a large amount of efforts into software development that might not be used among further discovery, and then how to scale accordingly.  

To see these diagrams as expected, download: https://chrome.google.com/webstore/detail/github-%2B-mermaid/goiiopgdnkogdbjmncgedmgpoajilohe/related and use Google Chrome. 

The following will be a suggestion for how I would deploy a **stable** ML Training Pipeline

```mermaid
graph LR

st[Start]-->rowop[Row Altering Operation]-->trainvaltest[Train/Val/Test Split Operation]-->colop[Columnar Operation]-->modeltraining[Machine Learning Model Training]-->e[End]

```

The following will be a suggestion for how I would deploy a **stable** ML Infernce Pipeline

```mermaid
graph LR

st[Start]-->rowop[Row Altering Operation]-->colop[Columnar Operation]-->modelinference[Machine Learning Model Inference]-->e[End]

```

