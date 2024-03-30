import pandas as pd
from matrix import MatrixFactorizationRecommender

train_data = pd.read_csv("./train_dataset.csv")
test_data = pd.read_csv("./test_dataset.csv")
print(train_data.shape, test_data.shape)

print("\nGradient Descent")
gd = MatrixFactorizationRecommender(model="gd", steps=100, lr=0.001)
gd.fit(train_data)
print(gd.evaluate_top_k(test_data))
print(gd.evaluate(test_data))
