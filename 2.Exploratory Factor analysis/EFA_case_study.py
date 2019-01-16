import factor_analyzer as fa
import pandas as pd
import utils as utl
import visual as vi
import pca.PCA as pca
import efa.EFA as efa


# Upload the input data file
file = "MortalityEU\\MortalityEU.csv"
table = pd.read_csv(file, index_col=0, na_values=':')
print(table)

obs_name = table.index
print(obs_name)
var_name = table.columns[1:]
print(var_name)

X = table[var_name].values
utl.replace_NA(X)
t = pd.DataFrame(X, index=obs_name, columns=var_name)
print(t)

pca = pca.PCA(X)
C = pca.getPrincipalComponents()
print("Principal components: ", C)
R = pca.getCorrelation()
alpha = pca.getEigenValues()
print("Eigenvalues: ", alpha)
a = pca.getEigenVectors()
print("Eigenvectors: ", a)

S, q, beta, common = efa.explore(C, alpha, R)
print("S: ", S)
print("q: ", q)
print("beta: ", beta)
print("common: ", common)

# Evaluate the information redundancy
# Bartlett test
Bartlett_test = fa.calculate_bartlett_sphericity(t)
print("Bartlett Test:", Bartlett_test)

# KMO test - Kaiser, Meyer, Olkin Measure Of Sampling Adequacy
kmo = fa.calculate_kmo(t)
print("Kaiser, Meyer, Olkin measure of sampling adequacy: ", kmo)
vi.correlogram(kmo[0], " KMO Indices")
print("KMO Total:", kmo[1])

if kmo[1] < 0.5:
    print("There is no any significant factor!")
    exit(1)

# Compose the model
fa_model = fa.FactorAnalyzer()
fa_model.analyze(t, rotation=None)

# Extract the factorial coefficients
loads = fa_model.loadings
loads.to_csv("Fa_Loadings.csv")
vi.correlogram(loads, "Factorial Coefficients")
vi.corrCircle(loads, 0, 1, "Factorial Coefficients - 1,2")
vi.corrCircle(loads, 0, 2, "Factorial Coefficients - 1,3")

# Apply factorial rotation
fa_model.analyze(t, rotation='varimax')
load_rot = fa_model.loadings
load_rot.to_csv("Fa_loadings_varimax.csv")
vi.correlogram(load_rot, "Factorial Coefficients - Varimax")
vi.corrCircle(load_rot, 0, 1, "Factorial Coefficients (Varimax) - 1,2")
vi.corrCircle(load_rot, 0, 2, "Factorial Coefficients (Varimax)- 1,3")

# Eigenvalues
eigvalues = fa_model.get_eigenvalues()
print(eigvalues)
