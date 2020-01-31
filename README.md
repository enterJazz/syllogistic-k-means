# syllogistic-k-means
A machine learning approach to human syllogistic reasoning using k-means

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
