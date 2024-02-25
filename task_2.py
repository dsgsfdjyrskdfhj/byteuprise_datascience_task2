
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


titanic_df = pd.read_csv('titanic.csv')


print(titanic_df.info())


print(titanic_df.describe())


print("Missing values per column:")
print(titanic_df.isnull().sum())


titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)



titanic_df['Cabin_Deck'] = titanic_df['Cabin'].astype(str).str[0]
titanic_df['Cabin_Deck'].fillna('Unknown', inplace=True)


titanic_df['Fare'] = np.log1p(titanic_df['Fare'])



titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
titanic_df['Embarked'] = titanic_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


plt.figure(figsize=(12, 8))
sns.barplot(x='Pclass', y='Survived', data=titanic_df, hue='Sex', ci=None)
plt.title('Survival Rate by Class and Gender')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()


plt.figure(figsize=(16, 10))
sns.boxplot(x='Pclass', y='Age', data=titanic_df, hue='Survived', palette='muted')
plt.title('Age Distribution by Class and Survival with Gender Hue')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()


correlation_matrix = titanic_df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


sns.pairplot(titanic_df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']])
plt.suptitle('Pairplot of Selected Variables', y=1.02)
plt.show()
