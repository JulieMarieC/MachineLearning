import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def convertir_fin_de_mois(yyyy_mm):
    return pd.to_datetime(yyyy_mm) + pd.offsets.MonthEnd(0)

if __name__ == "__main__":
    conso_elec = pd.read_excel('C:\\Users\\User\\PycharmProjects\\MachineLearning\\data\\eCO2mix_RTE_energie_M.xlsx')

    print(conso_elec.head())

    conso_elec = conso_elec[conso_elec['Territoire'] == "France"]
    conso_elec = conso_elec[['Mois', 'Consommation totale']]
    conso_elec.columns = ['mois', 'conso_totale']

    print(conso_elec.head())

    conso_elec.mois=[convertir_fin_de_mois(yyyy_mm) for yyyy_mm in conso_elec.mois]
    print(f"{conso_elec.head()}")

    conso_elec2 = conso_elec.set_index('mois')
    print(conso_elec2.head())

    plt.figure(figsize=(20,8))
    plt.plot(conso_elec2)
    plt.title("Consommation électrique - France")
    plt.xlabel("année/mois")
    plt.ylabel("GWh")
    plt.show()

    decomp_conso_elec = seasonal_decompose(conso_elec2.conso_totale, model='additive')
    decomp_conso_elec.plot()
    plt.show()

    # Création d'un échantillon d'entraînement
    mois_tronque = 12 # nombre de mois tronqués
    echantillon_train = conso_elec2[:-mois_tronque] # echantillon d'entraînement = échantillon étudié - 12 mois

    echantillon_test = conso_elec2[(len(conso_elec2)-12):len(conso_elec2)]

    print(f"echantillon test : {echantillon_test}")

    # Sur l'échantillon complet
    horizon_prevision_hw = 12
    periode_prevision_hw = pd.date_range(convertir_fin_de_mois(str(conso_elec2.index[len(conso_elec2)-1]+dt.timedelta(days=1))[0:7]),
                                         periods=horizon_prevision_hw,
                                         freq='M')
    hw = ExponentialSmoothing(np.asarray(conso_elec2.conso_totale), seasonal_periods=12, trend='add', seasonal='add').fit()
    hw_prediction_conso_corrige_dju = hw.forecast(horizon_prevision_hw)

    plt.figure(figsize=(20,6))
    plt.plot(conso_elec2.conso_totale, label='Consommation réalisée')
    plt.plot(periode_prevision_hw, hw_prediction_conso_corrige_dju, label='prédiction', color='red')
    plt.title("Consommation d'électricité - méthode Holt Winters")
    plt.xlabel('mois')
    plt.ylabel('GWh')
    plt.legend()
    plt.show()








