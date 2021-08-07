# Project 3: Machine Learning - Heart Attack Possibility

Overview:
1. Membuat model Machine Learning dengan algoritma Decision Tree dengan bahasa Python
2. Dataset berasal dari kaggle.com dengan nama 'Health care: Heart attack possibility Dataset', disusun oleh Naresh Bhat yang mengambil data dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
3. Dataset memiliki 14 kolom
    - **'age'** : age (umur)
    - **'sex'** : sex (jenis kelamin)
    - **'cp'** : chest pain type (4 values: 1 = Angina khas, 2 = Angina Atipikal, 3 = Nyeri non-Angina, 4 = Asimtomatik)
    - **'trestbps'** : resting blood pressure (Tekanan darah saat istirahat > 120 mg / dl)
    - **'chol'** : serum cholestoral in mg/dl
    - **'fbs'** : fasting blood sugar > 120 mg/dl
    - **'restecg'** : resting electrocardiographic results (values 0 = Normal ,1 = Memiliki gelombang kelainan ST-T, 2 = Menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri dengan criteria estes)
    - **'thalach'** : maximum heart rate achieved (detak jantung maksimum ynag dicapai)
    - **'exang'** : exercise induced angina (latihan diinduksi Angina)
    - **'oldpeak'** : oldpeak = ST depression induced by exercise relative to rest (Depresi ST akibat latihan relatif terhadap istirahat)
    - **'slope'** : the slope of the peak exercise ST segment (1 = condong ke atas, 2 = datar, 3 = sedikit landai)
    - **'ca'** : number of major vessels (0-3) colored by flourosopy (jumlah nadi utama)
    - **'thal'** : thal: 0 = normal; 1 = fixed defect; 2 = reversable defect (normal, cacat tetap, cacat sementara)
    - **'target'** : target: 0= less chance of heart attack 1= more chance of heart attack
4. Terdapat 7 tahapan dalam mengolah data dan membuat model, yaitu:
    - Import Libraries and Dataset
    - Remove Duplicates
    - Exploratory Data Analysis
    - Feature Engineering
    - Modeling
    - Model Evaluation
    - Save Model
5. Project menggunakan dataset berasal kaggle, disusun oleh Naresh Bhat. Dapat diakses [disini](https://www.kaggle.com/nareshbhat/health-care-data-set-on-heart-attack-possibility)

## 1. Import Libraries dan Dataset
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
```
```
df = pd.read_csv('./data/heart.csv')
```
```
df.head()
```
![image](https://user-images.githubusercontent.com/74480780/128595981-06a454ff-7d1d-498d-977e-275fa1b0f3ce.png)

```
df.info()
```
![image](https://user-images.githubusercontent.com/74480780/128596010-35da419f-8444-43a7-af98-27e96f171fd2.png)
```
df.shape
```
![image](https://user-images.githubusercontent.com/74480780/128596023-7381610d-dbf4-4e3b-baaf-b14dae86eaf3.png)

```
df.describe()
```
![image](https://user-images.githubusercontent.com/74480780/128596043-12657fee-b2d4-4507-8340-8e0cfed5f2a0.png)

- Check Null Value
```
df.isnull().sum()
```
![image](https://user-images.githubusercontent.com/74480780/128596064-65085004-e336-48a8-ad6d-72baf67f1979.png)

- Remove Duplicates
```
df.shape
```
![image](https://user-images.githubusercontent.com/74480780/128596084-dfa02980-0c33-405e-8358-4144077d301e.png)

```
df.loc[df.duplicated()]
```
![image](https://user-images.githubusercontent.com/74480780/128597060-1231f0b0-bd6d-438c-9083-3b95c3f3aece.png)

```
df = df.drop_duplicates(ignore_index=True)
```
```
df.shape
```
![image](https://user-images.githubusercontent.com/74480780/128596111-ab274ce3-be8a-4301-80d9-bf5c0523c046.png)

## 2. Exploratory Data Analysis
### 2.1 Melihat distribusi data
```
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_cols = ['sex' ,'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

num_colors = ['#FF4848', '#FFD371', '#64C9CF', '#9DDAC6', '#FA8072']
cat_colors = ['#64C9CF', '#C2FFD9', '#EFB7B7', '#FDE49C', '#28FFBF', '#FF67E7', '#FFF338', '#C9D8B6']
```
- Numerical Features
```
fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=800)
ax = ax.flatten()
plt.suptitle("Distribusi Feature Numerik dengan Histplot", fontsize=14, fontweight='bold')

for col, index in zip(num_cols, range(len(num_cols))):
    sns.histplot(ax=ax[num_cols.index(col)], data=df[col], color=num_colors[index])
    ax[num_cols.index(col)].grid(linewidth=0.5)
    ax[num_cols.index(col)].set_xlabel(f"'{col}'", fontsize=10, fontweight='bold')
    ax[num_cols.index(col)].set_ylabel("count", fontsize=10, fontweight='bold')

ax[len(num_cols)].set_axis_off()
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/128596183-ff8bbfea-9190-4f2b-aff2-816ad6bba390.png)

```
fig, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=800)
ax = ax.flatten()
plt.suptitle("Distribusi Feature Numerik dengan Boxplot", fontsize=14, fontweight='bold')

for col, index in zip(num_cols, range(len(num_cols))):
    sns.boxplot(ax=ax[num_cols.index(col)], data=df[col], orient='h', color=num_colors[index])
    ax[num_cols.index(col)].grid(linewidth=0.5)
    ax[num_cols.index(col)].set_xlabel(f"'{col}'", fontsize=10, fontweight='bold')

ax[len(num_cols)].set_axis_off()
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/128596203-182477ff-09d4-4215-8ba0-a0117b6aeb9c.png)

- Catagorical Features
```
fig, ax = plt.subplots(3, 3, figsize=(12, 6), dpi=800)
ax = ax.flatten()
plt.suptitle("Distribusi Feature Kategori dengan Countplot", fontsize=14, fontweight='bold')

for col, index in zip(cat_cols, range(len(cat_cols))):
    sns.countplot(ax=ax[cat_cols.index(col)], y= col, data=df, color=cat_colors[index])
    ax[cat_cols.index(col)].grid(linewidth=0.5)
    ax[cat_cols.index(col)].set_xlabel("count", fontsize=10, fontweight='bold')
    ax[cat_cols.index(col)].set_ylabel(f"'{col}'", fontsize=10, fontweight='bold')

ax[len(cat_cols)].set_axis_off()
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/128596227-fca01c27-f502-41a1-93d6-41a8c112e18d.png)

```
fig, ax = plt.subplots(2, 4, figsize=(12, 6), dpi=800)
ax = ax.flatten()
plt.suptitle("Distribusi Feature Kategori dengan Countplot berdasarkan Gender", fontsize=14, fontweight='bold')

for col, index in zip([col for col in cat_cols if col != 'sex'], range(len(cat_cols)-1)):
    sns.countplot(ax=ax[cat_cols.index(col)-1], x=col, data=df,
                  color=cat_colors[index],
                  hue='sex',
                  palette='hls')
    ax[cat_cols.index(col)-1].grid(linewidth=0.5)
    ax[cat_cols.index(col)-1].set_xlabel(f"'{col}'", fontsize=10, fontweight='bold')
    ax[cat_cols.index(col)-1].set_ylabel("count", fontsize=10, fontweight='bold')

# ax[len(cat_cols)].set_axis_off()
ax[len(cat_cols)-1].set_axis_off()
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/128596242-0893d410-8a52-4316-bbf2-42b625f63a95.png)

```
fig, ax = plt.subplots(2, 4, figsize=(8, 4), dpi=400)
ax = ax.flatten()
plt.suptitle("Distribusi Feature Kategori dengan Pie Chart", fontsize=10, fontweight='bold')

colors = ['#FFAAA7', '#867AE9', '#FFCEAD', '#C449C2', '#F5FDB0']

for col in cat_cols:
    df[col].value_counts().plot(ax=ax[cat_cols.index(col)], colors=colors ,kind='pie')
    ax[cat_cols.index(col)].set_xlabel(f"'{col}'", fontsize=7, fontweight='bold')
    ax[cat_cols.index(col)].set_ylabel(None)

plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/128596253-bfde0fab-3bde-421e-b461-dd5b177fa2e4.png)

### 2.2 Melihat korelasi data
```
fig, ax = plt.subplots(2, 2, figsize=(14, 7), sharey=True, dpi=800)
ax = ax.flatten()
plt.suptitle("Korelasi Umur terhadap Semua Feature Numerik berdasarkan Gender", fontsize=14, fontweight='bold')

for col, index in zip([col for col in num_cols if col != 'age'], range(len(num_cols)-1)):
    sns.scatterplot(ax=ax[num_cols.index(col)-1], x=col, y='age', data=df, hue='sex')
    ax[num_cols.index(col)-1].set_xlabel(f"'{col}'", fontsize=10, fontweight='bold')
    ax[num_cols.index(col)-1].set_ylabel(f"'age'", fontsize=10, fontweight='bold')
    ax[num_cols.index(col)-1].legend(['female', 'male'])

plt.ylim(ymax=90)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/128596277-0d445bea-db8f-4ed8-9bc4-be8e697a1177.png)

```
fig, ax = plt.subplots(figsize=(14, 4), dpi=600)
mask = np.triu(df[num_cols].corr())

sns.heatmap(df[num_cols].corr(), annot=True, cmap='Reds', mask=mask, linewidths=1)

plt.title("Korelasi Tiap Feature Numerik", fontsize=12, fontweight='bold')
plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/128596297-6133a0df-0857-4da0-af24-626a7e01eae2.png)

## 3. Feature Engineering
### 3.1 Membuat Kolom Baru dari Umur
Dengan rentang:

- kurang dari sama dengan 50 -> gol '1'
- 51 - 60 -> gol '2'
- 61 - 70 -> gol '3'
- lebih dari sama dengan 71 -> gol '4'
    
```
df['catage'] = pd.cut(df['age'], bins=[0, 50, 60, 70, 100], labels=[1, 2, 3, 4])
df.drop(columns='age', inplace=True)
```
```
df.head()
```
![image](https://user-images.githubusercontent.com/74480780/128596328-fe63eb46-2e4b-4657-b761-50702e48c03d.png)

## 4. Modeling
### 4.1 Splitting & Fitting
```
X = df.drop(columns='target')
y = df['target']
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```
![image](https://user-images.githubusercontent.com/74480780/128596367-54c9beb7-5913-4f1f-94fc-923c5b2bbd34.png)

```
print(f"Score: {model.score(X_test, y_test)}")
```
![image](https://user-images.githubusercontent.com/74480780/128596471-601e21ac-f506-4e6c-aea5-69c14718c528.png)

### 4.2 Classification Report
```
print(classification_report(y_test, model.predict(X_test)))
```
![image](https://user-images.githubusercontent.com/74480780/128596490-7c688860-152b-4dd1-9643-feeb24a78d54.png)

## 5. Model Evaluation
### 5.1 Hyperparameter Tuning dengan GridSearchCV
```
pipeline = Pipeline([
    ('algo', DecisionTreeClassifier())
])
```
- Tuning dua parameter pada DecisionTree yaitu parameter `criterion` dan `max_depth`
```
parameters = {
    'algo__max_depth':range(2, 21, 2),
    'algo__criterion':['gini', 'entropy']
}
```
```
new_model = GridSearchCV(pipeline, parameters, cv=4, verbose=1)
new_model.fit(X_train, y_train)
```
![image](https://user-images.githubusercontent.com/74480780/128596549-0b5bf2ae-56c9-4cca-93bf-7931db627650.png)

- Melihat parameter terbaik pada model baru
```
new_model.best_params_
```
![image](https://user-images.githubusercontent.com/74480780/128596565-1087e4f3-9cfd-4f6d-b648-e75cd0408c39.png)

- Membandingkan akurasi model pertama dan model kedua
```
model.score(X_test, y_test), new_model.score(X_test, y_test)
```
![image](https://user-images.githubusercontent.com/74480780/128596587-55f2849f-15fe-411d-9443-f12925ba8003.png)

- Melihat Classification Report dari model kedua
```
print(classification_report(y_test, new_model.predict(X_test)))
```
![image](https://user-images.githubusercontent.com/74480780/128596623-53578e87-33ce-49b9-bf9f-a32dd3c51db6.png)

### 5.2 Confusion Matrix
![](/images/cm_image.png)
*image by: MLeeDataScience in Toward Data Science*

- Membandingkan TRUE POSITIVE & TRUE NEGATIVE melalui Heatmap
```
mx_1 = confusion_matrix(y_test, model.predict(X_test))
mx_2 = confusion_matrix(y_test, new_model.predict(X_test))

fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200, sharey=True)
ax = ax.flatten()
sns.heatmap(mx_1, ax=ax[0], cmap='Blues', annot=True, linewidths=1)
sns.heatmap(mx_2, ax=ax[1], cmap='Blues', annot=True, linewidths=1)

plt.show()
```
![image](https://user-images.githubusercontent.com/74480780/128596893-e7833c3e-313b-4587-b1c6-b33edcd28425.png)

- Presentase prediksi model pertama
```
print(f"True Positif\t:{round(mx_1[0][0] / (mx_1[0][0] + mx_1[0][1]) * 100, 3)}%")
print(f"True Negatif\t:{round(mx_1[1][1] / (mx_1[1][1] + mx_1[1][0]) * 100, 3)}%")
```
![image](https://user-images.githubusercontent.com/74480780/128596923-0a5dc070-0a01-420a-8cd8-48a39bf246ed.png)

- Presentase prediksi model kedua
```
print(f"True Positif\t:{round(mx_2[0][0] / (mx_2[0][0] + mx_2[0][1]) * 100, 3)}%")
print(f"True Negatif\t:{round(mx_2[1][1] / (mx_2[1][1] + mx_2[1][0]) * 100, 3)}%")
```
![image](https://user-images.githubusercontent.com/74480780/128596940-e68091c9-b91c-4cd8-bb84-21b26518947f.png)

## 6. Save Model
```
import pickle
```
```
model_name = "heart.dt"
pickle.dump(model, open(model_name, 'wb'))
```
```
new_model_name = "new_heart.dt"
pickle.dump(new_model, open(new_model_name, 'wb'))
```

## Referensi

- Veratamala, Arinda. (2021). *Awas, Ini Akibatnya Jika Gula Darah Anda Terlalu Tinggi*. Diakses dari https://hellosehat.com/diabetes/akibat-gula-darah-tinggi/ pada 5 Agustus 2021.
- American Heart Association editorial staff. (2021). *Understanding Blood Pressure Readings*. Diakses dari https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings pada 6 Agustus 2021.
- American Heart Association editorial staff. (2021). *Target Heart Rates Chart*. Diaskes dari https://www.heart.org/en/healthy-living/fitness/fitness-basics/target-heart-rates pada 7 Agustus 2021.
- MLeeDataScience. (2021). *Visual Guide to the Confusion Matrix*. Diakses dari https://towardsdatascience.com/visual-guide-to-the-confusion-matrix-bb63730c8eba pada 7 Agustus 2021.
