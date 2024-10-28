from ucimlrepo import fetch_ucirepo
import tabulate
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
#print(wine_quality.variables) 
