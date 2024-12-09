import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def extract_num(obj):
    if not isinstance(obj, float):
        try:
            num = float(obj.split()[0])
        except ValueError:
            return pd.NA
        return num
    return obj


def preprocess_item(dict_item):
    df = pd.DataFrame(dict_item, index=[0])
    df['mileage'] = df['mileage'].apply(extract_num).astype('float')
    df['engine'] = df[df['engine'].notna()]['engine'].apply(
        extract_num
    ).astype('int')
    df['max_power'] = df['max_power'].apply(extract_num)
    df = df.drop('torque', axis=1)
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    dataset = pd.read_pickle('./dataset.pkl')
    X_cat = df.drop(['selling_price', 'name'], axis=1)
    temp = pd.concat([dataset, X_cat])
    X_cat = pd.get_dummies(temp, columns=[
        'fuel', 'transmission', 'owner', 'seller_type', 'seats'
    ], drop_first=True)
    X = pd.DataFrame(X_cat.iloc[[-1]])
    print(X)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def process_df(df):
    df = df.drop(['Unnamed: 0', 'selling_price', 'name'], axis=1)
    train_dataset = pd.read_pickle('./dataset.pkl')
    temp = pd.concat([train_dataset, df])
    X_cat = pd.get_dummies(temp, columns=[
        'fuel', 'transmission', 'owner', 'seller_type', 'seats'
    ], drop_first=True)
    X_test = X_cat.iloc[-1000:]
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_test))
    return X_test, X_scaled
