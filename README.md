# Machine-Learning-Algorithm

## Description

The `Machine-Learning-Algorithm.py` file is a Python script that implements a decision tree machine learning algorithm. The decision tree can be used for both classification and regression tasks. 

## How to Run the Code

You can run this code by executing the script in a Python environment. Make sure you have the necessary dependencies installed (see the "Dependencies" section below). You can run the script with the following command:

```
python Machine-Learning-Algorithm.py
```

**Note**: This command assumes that you have Python installed and that the script is in your current working directory.

## Dependencies

The script requires the following Python libraries:

- NumPy
- Pandas

You can install these dependencies using pip:

```
pip install numpy pandas
```

## Key Functionalities and Classes

The script primarily contains a class called `DecisionTree` that encapsulates the decision tree algorithm. The class includes methods for fitting the model to data (`fit`), predicting the target variable for given data points (`predict`), and several private methods used for constructing the tree.

The `DecisionTree` class uses a nested `Node` class to represent the nodes in the decision tree. Each `Node` can represent either a decision node (with a feature and a threshold for splitting) or a leaf node (with a value to be returned for predictions).

## Future Work

- **Pruning**: Implement pruning to avoid overfitting and reduce the complexity of the final decision tree.
- **Handling Missing Values**: Enhance the decision tree to handle missing values in the input data.
- **Support for Categorical Variables**: Currently, the decision tree only supports numerical variables. It could be extended to support categorical variables as well.
