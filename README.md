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

## Model Architectures

### Tensorflow Model

The `tf-mle` folder will have the example of how I would deploy a tensorflow model.  Any subclass of `tf.keras.models.Model` and `tf.keras.models.Sequential` model can be deploy using the method described in this section.  The main deployment method is to use Kubernetes to deploy specific `tensorflow/serving` web services, which will specify a particular tensorflow model.

We will also cover how to automate hyperparameter tuning in a way that will save the parameters.

### Scikit-Learn Model

The `mlflow-mle` folder will have the example of how I would deploy a scikit-learn model.


## Model Learning Development Flow

TODO

