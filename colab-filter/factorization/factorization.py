import pandas as pd
from matrix import MatrixFactorizationRecommender

train_data = pd.read_csv("./train_dataset.csv")

print("\nGradient Descent")
gd = MatrixFactorizationRecommender(model="gd", steps=100, lr=0.01)
gd.fit(train_data)
test_data = pd.read_csv("./test_dataset.csv")
print(gd.evaluate_top_k(test_data))
print(gd.evaluate(test_data))
