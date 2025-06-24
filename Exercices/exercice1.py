import numpy as np
import pandas as pd

if __name__ == '__main__':

    df = pd.DataFrame({
        'A': [np.nan, 3, 12],
        'B': [0,4,np.nan],
        'C': [2,np.nan, 4],
        'D': [np.nan,np.nan, 2]
    })

    print(df)

    df_sans_na = df.fillna(df.mean(numeric_only=True))
    print(df_sans_na)