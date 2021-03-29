import pandas as pd
from sklearn import preprocessing
from bayesnet import *

if __name__ == '__main__':
    data = pd.read_excel(r'C:\Users\maxim\OneDrive\Desktop\folder\diplom\data\parsing\final_companies.xlsx')

    df = data[['Wavelength_p', 'Energy_p',
               'Wavelength_f', 'Energy_f', 'Wavelength', 'Energy', 'Price', 'Shipment']]

    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled,  columns = ['Wavelength_p', 'Energy_p',
       'Wavelength_f', 'Energy_f', 'Wavelength', 'Energy', 'Price', 'Shipment'])

    for i, row in df.iterrows():
        w1 = row['Wavelength_p']
        e1 = row['Energy_p']
        w2 = row['Wavelength_f']
        e2 = row['Energy_f']
        w3 = row['Wavelength']
        e3 = row['Energy']
        dt = row['Shipment']
        ap = row['Price']

        BN = ProductBayesNet()
        BN.bake_network(e1, w1, e2, w2, e3, w3, dt, ap)
        score = BN.make_scores()
