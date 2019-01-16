import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as disc
import utils
import plots

f1 = 'ProiectBucuresti/ProiectB.csv'
f2 = 'ProiectBucuresti/ProiectBEstimare.csv'
t1 = pd.read_csv(f1, index_col=0)
t2 = pd.read_csv(f2, index_col=0)

utils.replace_na_df(t1)
utils.replace_na_df(t2)

var = np.array(t1.columns)
print(var)
var_categorical = var[0:6]
print(var_categorical)
utils.coding(t1, var_categorical)
utils.coding(t2, var_categorical)
print(t1)
print(t2)

# Select the predictor variables and the discriminant variable
var_p = var[0:11]
var_c = "VULNERAB"
print(var_p)
print(var_c)

x = t1[var_p].values
print(x)
y = t1[var_c].values
print(y)

# Build the model
lda_model = disc.LinearDiscriminantAnalysis()
lda_model.fit(x, y)

class_setBase = lda_model.predict(x)
table_classificationB = pd.DataFrame(
    data={str(var_c[0]): y, 'prediction': class_setBase},
    index=t1.index)
table_classificationB.to_csv('ClassB.csv')
tabel_clasificareB_err = table_classificationB[y != class_setBase]
tabel_clasificareB_err.to_csv('ClassB_err.csv')
n = len(y)
n_err = len(tabel_clasificareB_err)
degree_of_credence = (n - n_err) * 100 / n
print("Degree of credence:", degree_of_credence)

# Apply on the test set
class_setTest = lda_model.predict(t2[var_p].values)
table_of_classification = pd.DataFrame(
    data={'prediction': class_setTest},
    index=t2.index
)
table_of_classification.to_csv("ClassificationT.csv")

# Compute group accuracy
g = lda_model.classes_
q = len(g)
mat_c = pd.DataFrame(data=np.zeros((q, q)), index=g, columns=g)
for i in range(n):
    mat_c.loc[y[i], class_setBase[i]] += 1
accuracy_groups = np.diag(mat_c) * 100 / np.sum(mat_c, axis=1)
mat_c['Accuracy'] = accuracy_groups
mat_c.to_csv("Accuracy.csv")

# print(mat_c)
# Instances on the first 2 axes of discrimination
u = lda_model.scalings_
z = x @ u
xc = lda_model.means_
zc = xc @ u

# Get the number of discriminant axes (number of columns in u)
r = np.shape(u)[1]
if r > 1:
    plots.scatter_discriminant(z[:, 0], z[:, 1], y, t1.index, zc[:, 0], zc[:, 1], g)
for i in range(r):
    plots.distribution(z[:, i], y, g, axa=i)
plots.show()
