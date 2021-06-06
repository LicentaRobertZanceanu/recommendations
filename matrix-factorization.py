import pandas as pd
import numpy as np
from matrix_factorization import BaselineModel, KernelMF, train_update_test_split
from sklearn.metrics import mean_squared_error

ratingCsv = pd.read_csv('rating.csv')
ratingCsv.rename(columns={
    'genre._id': 'genreId',
    'artist._id': 'artistId',
    'userId':'user_id',
    'songId':'item_id',
}, inplace=True)

print(ratingCsv)
X = ratingCsv[['user_id', 'item_id']]
Y = ratingCsv['rating']

(
    X_train_initial,
    y_train_initial,
    X_train_update,
    y_train_update,
    X_test_update,
    y_test_update,
) = train_update_test_split(ratingCsv, frac_new_users=0.2)

matrix_fact = KernelMF(n_epochs=100, n_factors=100, verbose=1, lr=0.001, reg=0.005)
matrix_fact.fit(X_train_initial, y_train_initial)

pred = matrix_fact.predict(X_test_update)
rmse = mean_squared_error(y_test_update, pred, squared=False)
print(f"\nTest RMSE: {rmse:.4f}")

user = '60a032a282d27c42cc2d7146'
items_known = X_train_initial.query("user_id == @user")["item_id"]
matrix_fact.recommend(user=user, items_known=items_known)