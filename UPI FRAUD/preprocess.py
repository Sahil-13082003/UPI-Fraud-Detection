from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Consistent encoders (these should match the ones used in training)
type_encoder = LabelEncoder()
type_encoder.classes_ = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

def custom_hash(value):
    return hash(value) % 10**6  # Match the int32 encoding used before

def preprocess_input(df):
    df['type'] = type_encoder.transform(df['type'])

    df['nameOrig'] = df['nameOrig'].apply(custom_hash).astype('int32')
    df['nameDest'] = df['nameDest'].apply(custom_hash).astype('int32')

    return df[[
        'step', 'type', 'amount', 'nameOrig',
        'oldbalanceOrg', 'newbalanceOrig', 'nameDest',
        'oldbalanceDest', 'newbalanceDest'
    ]]
