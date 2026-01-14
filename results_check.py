import pickle as pkl
import pandas as pd

with open('output/SSVEP/1-1/04122024_005613/test_output.pkl','rb') as f:
    object = pkl.load(f)

df = pd.DataFrame(object)
df.to_csv(r'output_test6.csv')

#data = pkl.load(open('output/SSVEP/1-1/03122024_072659/test_output.pkl','rb'))


