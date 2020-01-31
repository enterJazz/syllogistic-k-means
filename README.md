# syllogistic-k-means
A machine learning approach to human syllogistic reasoning using k-means++

## What are syllogisms?
Syllogisms are simple reasoning tasks, which serve as a framework for human reasoning.
A syllogism consists of two quantified statements, e.g.:
"All Artists are Bakers" and "No Bakers are Dancers", or 
"Some Policemen are Firefighters" and "Some Firefighters are not Cowboys".

A human response to a syllogism, such as the one above, might be:
"Some Policemen are not Cowboys"
These human responses may, of course, not be logically valid (i.e. logically correct).

The program in this repository tries to predict these human syllogisms for each individual participant, adapting to the individuals answers.

## How to use
### Prerequisites
Following packages are required to run syllogistic-k-means:
- ccobra: `pip install ccobra` (for more info see https://github.com/CognitiveComputationLab/ccobra)
- pandas: `pip install pandas`
- numpy: `pip install numpy`

### Execution
To execute, enter the following command in terminal in the project directory:

`ccobra kmeans.json`

After that, the program will compute a k-means clustering based on the given training data (specified in the `.json`) and use this clustering to predict human responses to syllogistic problems, using the test data! How cool is that?
The results will then be displayed in the browser of your choice with `html`. Give it a try!

### Configuration

Natively, this approach uses k = 6. To change this parameter for the k-means clustering, edit the k parameter in `/models/cluster_extractor/syllogistic_kmeans_model.py`.

In the `kmeans.json`, there is an option to activate cross validation. To activate this, set the cross validation boolean to true. This will make the model take a lot longer, but will generate a better accuracy.

To see the results of the clusters, what participants chose what cluster and can be described using which syllogistic principle, `cd` to  `/models/cluster_extractor` and execute
`python generate_results.py (#number of clusters / k)`, so e.g. `python generate_results.py 5` would generate results using k-means with k = 5.
These results are written as `.csv` in `./results`
