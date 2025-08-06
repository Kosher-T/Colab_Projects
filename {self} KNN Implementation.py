# Import necessary libraries
import numpy as np                           # For numerical operations and array handling
import pandas as pd                          # For data manipulation and analysis
import matplotlib.pyplot as plt              # For data visualization (plotting histograms)
from sklearn.preprocessing import StandardScaler   # For feature scaling (standardization)
from imblearn.over_sampling import RandomOverSampler   # For handling class imbalance by oversampling
from collections import Counter              # For counting occurrences of elements in a list

# Define column names for the dataset
cols = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"
]

# Load the MAGIC gamma telescope dataset into a pandas DataFrame
# The data is expected to be in the specified path with columns named as above
df = pd.read_csv("C:\\Code\\Code\\.Files\\magic04.data", names=cols)

# Convert the class column to binary: 1 if 'g' (gamma), 0 otherwise (hadron)
df["class"] = (df["class"] == 'g').astype(int)

# Plot histograms for each feature, separated by class (gamma vs hadron)
for label in cols[:-1]:  # Exclude the 'class' column itself
    plt.hist(
        df[df["class"] == 1][label],
        color='blue',
        label='gamma',
        alpha=0.7,
        density=True
    )
    plt.hist(
        df[df["class"] == 0][label],
        color='red',
        label='hadron',
        alpha=0.7,
        density=True
    )
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

# Randomly shuffle and split the dataset into train (60%), validation (20%), and test (20%) sets
train, valid, test = np.split(
    df.sample(frac=1),  # Shuffle the dataset
    [int(0.6*len(df)), int(0.8*len(df))]
)

def scale_dataset(dataframe, oversample=False):
    """
    Standardizes features and optionally applies random oversampling for class balance.
    
    Parameters:
        dataframe (DataFrame): Input data including features and class label.
        oversample (bool): Whether to balance classes using random oversampling.
        
    Returns:
        data (ndarray): Combined scaled features and labels.
        X (ndarray): Scaled feature matrix.
        y (ndarray): Target vector.
    """
    # Separate features and target
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    
    # Standardize features to zero mean and unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # fit gets the mean and std, transform applies the formula (value - mean) / std to each feature
    
    # Optionally balance classes using random oversampling
    if oversample:
        rods = RandomOverSampler()
        X_res, y_res = rods.fit_resample(X, y)  # fit finds the value in (y) that appears the least number of times, resample duplicates random
                                     # rows of the least appearing value until the len matches the most appearing value, making them equal

    # Combine features and target for convenience
    data = np.hstack((X, np.reshape(y, (-1, 1))))  # np.hstack() takes one argument, a tuple of arrays to stack horizontally
                                                   # np.reshaped turns our horizontal y into a vertical column vector to match the shape of X
                                                   # Combined, they simply create an array where y is just another column at the end of X

    return data, X, y   # Return the processed data


train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)


# First step, Euclidean distance function
def euclid(point_a, point_b):
    squared_diffs = (point_a - point_b)**2
    summed = np.sum(squared_diffs)
    return np.sqrt(summed)
  # Very straightforward and easy. Able to cater to lists of any length

# Second step, KNN function
def k_nearest_optimized(point, train_features, train_labels, k=5):
    # Need to understand this part especially well
    # 1. Calculate all distances at once (super fast)
    distances = np.linalg.norm(train_features - point, axis=1)

    # 2. Get the indices of the 'k' smallest distances
    k_nearest_indices = np.argsort(distances)[:k]

    # 3. Get the labels (allegiances) of those neighbors
    k_nearest_allegiances = train_labels[k_nearest_indices]
    
    # You could also return the points themselves if needed
    k_nearest_points = train_features[k_nearest_indices]

    return k_nearest_points, k_nearest_allegiances

# Third step, KNN prediction function
def classifier(point, X, y):
    neighbours, allegiances = k_nearest_optimized(point, X_train, y_train)
    # need to find the max votes. I will use Counter. Should help refresh my memory
    count = Counter(allegiances)
    most_common_item = count.most_common(1)
    prediction = most_common_item[0][0]

    return prediction


# Example usage of the classifier
# Let's classify a random point from the validation set
print(classifier(X_valid[0], X_train, y_train))