import pandas as pd
from imblearn.over_sampling import RandomOverSampler

#قراءة البيانات و القاء نظرة سريعة
df = pd.read_csv('datasets/Churn_Modelling.csv')
print(df.columns)
print(df.info)
value_counts = df['Exited'].value_counts()
print(value_counts)

#تصحيح العمود exited
input_cul = df.drop('Exited',axis=1)
class_cul = df['Exited']

RS = RandomOverSampler(random_state=0)

input_cul_R,class_cul_R = RS.fit_resample(input_cul,class_cul)
df_blansed = pd.concat([input_cul_R,class_cul_R],axis=1)

print(df_blansed['Exited'].value_counts())


#تقسيم البيانات
X = df_blansed.iloc[:,3:13].values
Y = df_blansed.iloc[:,13].values