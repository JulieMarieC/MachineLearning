import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

if __name__ == '__main__':

    # Importer jeu de données
    csv_folder = "C:\\Users\\User\\PycharmProjects\\MachineLearning\\data"
    customers_file_name = "db_customers.csv"
    customers_file = os.path.join(csv_folder, customers_file_name)
    df = pd.read_csv(customers_file)

    print(df)
    print(df.describe())
    print(df.shape)
    print(df.columns)

    # Sélectionner des données
    print(df.head())
    print(df['order_status'].value_counts())
    df = df.drop('payment_installments', axis="columns")
    print(df)
    df2 = df[['customer_unique_id', 'review_score']]
    print(df2)

    # Identifier et traiter les valeurs manquantes
    print(df.isnull().sum())

    df_valeurs_manquantes = df[df['order_delivered_customer_date'].isna()]
    print(df_valeurs_manquantes)

    df_delivered = df[df['order_delivered_customer_date'].notna()]
    print(df_delivered.isnull().sum())

    df_filled = df_delivered.fillna({'product_category_name_english': 'Non renseigné', 'customer_city': 'Inconnu'})
    print(df_filled.isnull().sum())
    print(df_filled)

    # Effectuer une jointure
    reviews_file_name = "db_reviews.csv"
    reviews_file = os.path.join(csv_folder, reviews_file_name)
    df2 = pd.read_csv(reviews_file)
    print(df2.head())

    dataframe_total = pd.merge(df, df2, on='order_id', how = 'inner')
    print(dataframe_total.head())

    print(len(df['order_id'].unique()))
    print(len(dataframe_total['order_id'].unique()))

    # Grouper et agréger les données
    df_grouped = df.groupby(['customer_unique_id', 'customer_state']).agg({'payment_value' : 'sum', 'order_id' : 'count'}).reset_index()
    print(df_grouped.head())

    # Normaliser les données
    colonnes_numeriques = df.select_dtypes(include=['int64', 'float64']).columns
    print(colonnes_numeriques)

    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=['int64', 'float64'])), columns=df.select_dtypes(include=['int64', 'float64']).columns)
    print(df_normalized.head())

    scaler2 = StandardScaler()
    df_standardized = pd.DataFrame(scaler2.fit_transform(df.select_dtypes(include=['int64', 'float64'])), columns=df.select_dtypes(include=['int64', 'float64']).columns)
    print(df_standardized.head())

    # Traiter les valeurs catégorielles
    # 1 One Hot Encoding / Get Dummies Encoding
    encoded_status = pd.get_dummies(df, columns=['order_status'])
    print(encoded_status.head())

    # 2 Label Encoding
    label_encoder = LabelEncoder()
    df['product_category_name_english'] = label_encoder.fit_transform(df['product_category_name_english'])
    print(df.head())