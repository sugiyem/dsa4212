import pandas as pd
from matrix_factorization import MatrixFactorizationRecommender

train_data = pd.read_csv("./train_dataset.csv")
test_data = pd.read_csv("./test_dataset.csv")
print(train_data.shape, test_data.shape)

print("\nGradient Descent")
gd = MatrixFactorizationRecommender(model="gd")
gd.fit(train_data)
print(gd.evaluate_top_k(test_data))

print("\nStochastic Gradient Descent")
sgd = MatrixFactorizationRecommender(model="sgd")
sgd.fit(train_data)
print(sgd.evaluate_top_k(test_data))

print("\nSingular Value Decomposition")
svd = MatrixFactorizationRecommender(model="svd")
svd.fit(train_data)
print(svd.evaluate_top_k(test_data))

print("\nAlternating Least Squares")
als = MatrixFactorizationRecommender(model="als")
als.fit(train_data)
print(als.evaluate_top_k(test_data))

print("\nAlternating Least Squares (np.solve)")
als_solve = MatrixFactorizationRecommender(model="als_solve")
als_solve.fit(train_data)
print(als_solve.evaluate_top_k(test_data))
