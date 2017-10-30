"""
The :mod:`data_tools` module contains data processing tools
for use in the titanic problem
"""
import pandas as pd


def prepare_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Selecting data we care about and cleaning it up"""
    data_to_use = raw_data[['Survived', 'Pclass', 'Fare', 'Sex', 'Age', 'Name']]
    data_to_use = data_to_use.dropna(axis=0)

    data_to_use = data_to_use.replace(['male', 'female'], [1, 0])

    data_to_use['Prefix'] = data_to_use['Name'].apply(extract_prefix)
    data_to_use = data_to_use.drop('Name', axis=1)

    data_to_use = pd.get_dummies(data_to_use)

    return data_to_use


def extract_prefix(name: str) -> str:
    """Get a person's title from a name formatted 'Last, Title. First'"""
    return name[(name.find(', ') + len(', ')):name.find('.')]
