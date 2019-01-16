import graphics
import pandas as pd
import PCA


#read input data from csv file
table = pd.read_csv("Teritorial.csv",index_col=0)

#bring the table data into a numpy matrix (ndarray)
X = table.iloc[:, 1:].values
obs_name = table.index
var_name = table.columns[1:]

n = X.shape[0] #number of observation
m = X.shape[1] #number of variables
#print(n, m)

#invoke pca function
R, alpha, a, Rxc, C = PCA.pca(X)
#print(R)

#save the processed table
X_Tab = pd.DataFrame(data=X, columns=var_name, index=obs_name)
X_Tab.to_csv(path_or_buf="X.csv")

#save the correlation matrix
R_Tab = pd.DataFrame(data=R, columns=var_name, index=var_name)
R_Tab.to_csv(path_or_buf="R.csv")

#show the correlogram
graphics.correlogram(R_Tab)

#save the correlation factors
Rxc_Tab = pd.DataFrame(Rxc, index=var_name, columns=
                       ["C"+str(k) for k in range(m)])
Rxc_Tab.to_csv("Rxc.csv")

#show factors correlogram
graphics.correlogram(Rxc_Tab, "Correlogram of the factors")

#show the eigenvalues graphic
graphics.eighenValues(alpha)
