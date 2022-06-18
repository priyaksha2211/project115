import pandas as pd
import plotly.express as px
df = pd.read_csv("escape_velocity.csv")
Velocity = df["Velocity"].tolist()
Escaped=df["Escaped"].tolist()
fig = px.scatter(x=Velocity,y=Escaped)
fig.show()


import numpy as np
temperature_array=np.array(Velocity)
melted_array=np.array(Escaped)

m,c = np.polyfit(temperature_array,melted_array,1)

y=[]
for x in temperature_array:
  y_value = m*x+c
  y.append(y_value)

fig=px.scatter(x=temperature_array,y=melted_array)
fig.update_layout(shapes=[
                          dict(
                              type='line',
                              y0=min(y),
                              y1=max(y),
                              x0=min(temperature_array),
                              x1=max(temperature_array)
                          )
])
fig.show()




import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X=np.reshape(Velocity,(len(Velocity), 1))
Y=np.reshape(Escaped,(len(Escaped), 1))

lr=LogisticRegression()
lr.fit(X,Y)

plt.figure()
plt.scatter(X.ravel(), Y, color='grey', zorder=20)

def model(x):
  return 1 / (1 + np.exp(-x))

X_test=np.linespace(0,100,200)
melting_chances=model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, melting_chances, color='cyan', linewidth=3)
plt.axhline(y=0, color='red', linestyle='-')
plt.axhline(y=1, color='red', linestyle='-')
plt.axhline(y=0.5, color='cadetblue', linestyle='-')

plt.axvline(x=X_test[23], color='green', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(0,30)
plt.show()
print(X_test[23])