# Notes

## IDataView

Represents data in ML and is known for:

- Immutability
Does not change root data in any way. Actions performed on the data is repeatable without the need to clone the root data.
- Lazy 
Components should be lazy through their rows and columns.  With only a subset of data that is requested, computation on other columns should be avoided.
- High dimensionality 
Set of primitive values can be grouped together in a single value vector column.


## DataViewSchema

Schema of IDataView which provides the types, names and other annotations that make up an IDataView. Before loading your data, this has to be created through POCO's or classes.

## Support for loading data

- Files ( text, binary, images  ) | single or multiple files
- DB ( sqllite, ms sql, postgress, mysql, )
- Enumerable ( eg. json, XML, ...)

https://docs.microsoft.com/nl-be/dotnet/machine-learning/how-to-guides/load-data-ml-net

```
public class HousingData
{
    [LoadColumn(0)]
    public float Size { get; set; }

    [LoadColumn(1, 3)]
    [VectorType(3)]
    public float[] HistoricalPrices { get; set; }

    [LoadColumn(4)]
    [ColumnName("Label")]
    public float CurrentPrice { get; set; }
}
```

The above defines a file, probably a csv. Column 0 = the size. 1-3 contains prices and the current price is found in the 4th column.

What we want to predict, is known as the label. 


# Model training

3 steps:
1) Preparing your data
2) Choosing an algorithm
3) Training the model


## 1. Preparing your data

Splits the data with 20% of test.
```
 var trainTestSplit = mlContext.Data.TrainTestSplit(trainingData, testFraction: 0.2);
```

OneHotEncoding makes the combination unique.
The features column is normalized into a minmax value with 0 - 1.

```
// Create data transformation pipeline
var dataProcessPipeline =
    mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MakeEncoded", inputColumnName: "Make")
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModelEncoded", inputColumnName: "Model"))
        .Append(mlContext.Transforms.Concatenate("Features", "Year", "Mileage", "MakeEncoded", "ModelEncoded"))
        .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
        .AppendCacheCheckpoint(mlContext);
```

Then a suitable algorithm is picked : https://docs.microsoft.com/nl-be/dotnet/machine-learning/how-to-choose-an-ml-net-algorithm
```
  // Choose an algorithm and add to the pipeline
            var trainer = mlContext.Regression.Trainers.LbfgsPoissonRegression();
            var trainingPipeline = dataProcessPipeline.Append(trainer);
```

- Text classification:
  - Averaged perceptron
- Large number of features
  - L-BFGS
- Recommendation engines
  - Matrix Factorization
- Clustering
  - K-Means
- Anomely detection
  - Principal component analysis ( PCA trainer)
- Multi class classification on a small dataset
  - Naive Bayes
- Baseline performance of other trainers
  - Prior trainer

  Vector datasets : to check


## Train the model
= Fitting

```
// Train the model
Console.WriteLine("Training model...");
var model = trainingPipeline.Fit(trainTestSplit.TrainSet);
```

## Evaluate the mdoel
https://docs.microsoft.com/nl-be/dotnet/machine-learning/resources/metrics
Makes predications on both the train & test sets.

```
  // Make predictions on train and test sets
            IDataView trainSetPredictions = model.Transform(trainTestSplit.TrainSet);
            IDataView testSetpredictions = model.Transform(trainTestSplit.TestSet);
```


Evaluate compares label & score columns for both datasets. 

```
// Calculate evaluation metrics for train and test sets
var trainSetMetrics = mlContext.Regression.Evaluate(trainSetPredictions, labelColumnName: "Label", scoreColumnName: "Score");
var testSetMetrics = mlContext.Regression.Evaluate(testSetpredictions, labelColumnName: "Label", scoreColumnName: "Score");
```

And then a score can be generated
```
Console.WriteLine($"Train Set R-Squared: {trainSetMetrics.RSquared} | Test Set R-Squared {testSetMetrics.RSquared}");
```

This results in a value Eg.
Train Set R-Squared: 0,8947136495829634 | Test Set R-Squared 0,8978774136439497

The test set is higher than the train set. This means that there is no overfitting happening and means we have a good result.

## Cross validation ( optional) 
Is a training and evaluation technique. 

Helps with overfitting, by training multiple models on parts of the data and keeps some data out of the training process.

```
using System.Linq;
var crossValidationResults = mlContext.Regression.CrossValidate(trainingData, trainingPipeline, numberOfFolds: 5);
var avgRSquared = crossValidationResults.Select(model => model.Metrics.RSquared).Average();
Console.WriteLine($"Cross Validated R-Squared: {avgRSquared}");
...

This value should be smaller than the previous value. Which seems correct:

Cross Validated R-Squared: 0.8736620547207405

##Save the model
```
// Save model
Console.WriteLine("Saving model...");
mlContext.Model.Save(model, trainingData.Schema, MODEL_FILEPATH);
```


# Using the model

While we used Transforms for training the model. We now want to create a single predication. For multi-thread environments you should use the 'PredictionEnginePool ' service and for a single prediction, you can use the "PredictionEngine'.

PredictionEnginePool can easily be used in asp.net

- Install the 'Microsoft.Extensions.ML' nuget

Add 
```
services.AddPredictionEnginePool<ModelInput, ModelOutput>().FromFile(modelName:"PricePrediction",filePath:@"C:\Dev\MLModel.zip");
```
