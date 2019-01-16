import sklearn.cross_decomposition as skl
import pandas as pd
import numpy as np
import utils
import visual

# Import 2 data sets from a single csv file
input_file = "EnergieEU\\Energy.csv"
table = pd.read_csv(input_file, index_col=0)

obs_name = table.index
print(obs_name)
var_name = table.columns[1:]
print(var_name)

x_columns = var_name[:4]
y_columns = var_name[4:]
print(x_columns, y_columns)

X = table[x_columns].values
print(X)
Y = table[y_columns].values
print(Y)
# utils.replace_NA(X)
# utils.replace_NA(Y)

x_df = pd.DataFrame(X, index=obs_name, columns=x_columns)
print(x_df)
x_df.to_csv("X_dataset.csv")
y_df = pd.DataFrame(Y, index=obs_name, columns=y_columns)
print(y_df)
y_df.to_csv("Y dataset.csv")


# Build the model
n, p = np.shape(X)
print("n = ", n)
print("p = ", p)
q = np.shape(Y)[1]
print("q = ", q)
m = min(p, q)
print("m = ", m)


cca_model = skl.CCA(n_components=m)
cca_model.fit(X, Y)

# Get the results
# Cononical scores
z = cca_model.x_scores_
print("z canonical scores: ", z)
z_df = pd.DataFrame(z)
print(z_df)
# z.tofile("z.csv", ',')
z_df.to_csv("z.csv", sep=',')


u = cca_model.y_scores_
print("u canonical scores: ", u)
u_df = pd.DataFrame(u)
print(u_df)
# u.tofile("u.csv", ',')
u_df.to_csv("u.csv", sep=',')

print("X factor loadings: ", cca_model.x_loadings_)
print("Y factor loadings: ", cca_model.y_loadings_)

# Compute the canonical correlations
r_list = [np.corrcoef(z[:, i], u[:, i], rowvar=False)[0, 1] for i in range(m)]
# print(r_list)
r = np.array(r_list)
print("r = ", r)

chi2_computed, chi2_estimated = utils.bartlett_wilks(r, n, p, q, m)
print("Chi square computed : ", chi2_computed)
print("Chi square test: ", chi2_estimated)

chi2_computed_table = pd.DataFrame(chi2_computed, index=['r' + str(i) for i in range(1, m + 1)], columns=['chi2_computed'])
print("Chi square computed table: ", chi2_computed_table)
visual.correlogram(chi2_computed_table, "Bartlett-Wilks significance test", 0)

chi2_estimated_table = pd.DataFrame(chi2_estimated, index=['r' + str(i) for i in range(1, m + 1)], columns=['chi2_estimated'])
print("Chi square estimated table: ", chi2_estimated_table)
visual.correlogram(chi2_estimated_table, "Bartlett-Wilks significance test", 0)


