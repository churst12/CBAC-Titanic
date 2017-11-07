"""
The :mod:`data_tools` module contains data processing tools
for use in the titanic problem
"""
import pandas as pd


def prepare_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Selecting data we care about and cleaning it up"""
    data_to_use = raw_data[['Survived', 'Pclass', 'Fare', 'Sex', 'Age', 'Name', 'SibSp', 'Parch']]
    data_to_use.is_copy = False
    data_analysis(data_to_use)

    # Family size
    data_to_use['FamilySize'] = data_to_use['SibSp'] + data_to_use['Parch'] + 1

    # Binary Sex
    data_to_use['Sex'] = data_to_use['Sex'].map({'female': 0, 'male': 1})

    # Age imputation
    data_to_use['Age'] = data_to_use['Age'].fillna(data_to_use['Age'].mean())

    # Extracting prefixes
    data_to_use['Prefix'] = data_to_use['Name'].apply(extract_prefix)
    data_to_use = data_to_use.drop('Name', axis=1)

    rare_prefixes = ['Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    data_to_use['Prefix'] = data_to_use['Prefix'].replace(rare_prefixes, 'Rare')
    data_to_use['Prefix'] = data_to_use['Prefix'].replace('Mlle', 'Miss')
    data_to_use['Prefix'] = data_to_use['Prefix'].replace('Ms', 'Miss')
    data_to_use['Prefix'] = data_to_use['Prefix'].replace('Mme', 'Mrs')
    data_to_use['Prefix'] = data_to_use['Prefix'].map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare': 4})

    data_analysis(data_to_use)

    return data_to_use.dropna()


def extract_prefix(name: str) -> str:
    """Get a person's title from a name formatted 'Last, Title. First'"""
    return name[(name.find(', ') + len(', ')):name.find('.')]


def data_analysis(data: pd.DataFrame):
    # print(data.head())
    print(data.info())
    corr_matrix = data.corr()
    # print(corr_matrix['Age'])
