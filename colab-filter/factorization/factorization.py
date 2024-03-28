import pandas as pd
from matrix import MatrixFactorizationRecommender

train_data = pd.read_csv("./train_dataset.csv")
test_data = pd.read_csv("./test_dataset.csv")

gd = MatrixFactorizationRecommender(model="svd", k=10)
gd.fit(train_data)
print(gd.predict(test_data))
