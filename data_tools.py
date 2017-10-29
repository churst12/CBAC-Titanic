import pandas as pd


def prepare_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    # Selecting data we care about and cleaning it up
    data_to_use = raw_data[["Survived", "Pclass", "Fare", "Sex", "Age", "Name"]]  # these columns will be included in training data
    non_null_data = data_to_use.dropna(axis=0)  # drop rows that contain null value

    two_genders_data = non_null_data.replace(["male", "female"], [1, 0])

    names = two_genders_data["Name"].values
    name_prefixes = pd.Series((name[(name.find(", ") + len(", ")):name.find(".")] for name in names), name="Prefix")
    prefixed_data = two_genders_data.join(name_prefixes).drop("Name", axis=1)

    one_hot_data = pd.get_dummies(prefixed_data).astype(int)

    return one_hot_data
