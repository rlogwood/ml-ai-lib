import pandas as pd
import class_imbalance as ci

def test_class_imbalance():
    # Load the dataset
    df = pd.read_csv('./test/data/loan_data.csv')
    # explore data
    #print(df)
    target_col = 'not.fully.paid'
    result = ci.check_imbalance(df[target_col].value_counts())
    #assert result == False


