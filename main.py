import pandas as pd
from sklearn import preprocessing
from bayesnet import *
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def make_net_scores():
    data = pd.read_excel(r'C:\Users\maxim\OneDrive\Desktop\folder\diplom\data\parsing\final_companies_with_text.xlsx')

    df = data[['Wavelength_p', 'Energy_p',
               'Wavelength_f', 'Energy_f', 'Wavelength', 'Energy', 'Price', 'Shipment', 'Total_x', 'Total_y',
               'Total_medicine', 'In_spie', 'Country', 'CB', 'Public/private', 'aws1', 'aws2', 'aws3', 'aws4',
               'Mensions', 'Positiveness', 'Average positiveness']]

    df['Totalxy'] = df['Total_x'] + df['Total_y']
    df['Country_good'] = np.where(df['Country'].isin(['finland', 'us', 'china', 'germany']), 1, 0)
    df['Shares'] = np.where(df['Public/private'] == 'public', 1, 0)
    df['aws1'] = np.where(df['aws1'] == 'None', df['aws1'].mean(), df['aws1'])
    df['aws2'] = np.where(df['aws2'] == -1, df['aws2'].mean(), df['aws2'])
    df['aws3'] = np.where(df['aws3'] == -1, df['aws3'].mean(), df['aws3'])
    df['aws4'] = np.where(df['aws4'] == -1, df['aws4'].mean(), df['aws4'])

    df.drop(columns=['Total_x', 'Total_y', 'Country', 'Public/private'], inplace=True)

    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=['Wavelength_p', 'Energy_p', 'Wavelength_f', 'Energy_f', 'Wavelength',
                                         'Energy', 'Price', 'Shipment', 'Total_medicine', 'In_spie', 'CB',
                                         'aws1', 'aws2', 'aws3', 'aws4', 'Mensions', 'Positiveness', 'Average positiveness',
                                         'Totalxy', 'Country_good', 'Shares'])

    df['Url'] = data['Url1']
    df['target'] = data['CB']
    data_scores = {}

    for i, row in df.iterrows():
        # Product Net
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
        score_product = BN.make_scores()

        # Tech Net
        e1 = row['Total_medicine']
        w1 = row['Totalxy']
        e2 = row['In_spie']
        BN = TechBayesNet()
        BN.bake_network(e1, w1, e2)
        score_tech = BN.make_scores()

        # Org Net
        e1 = row['Country_good']
        w1 = row['Shares']
        e2 = row['CB']
        BN = OrgBayesNet()
        BN.bake_network(e1, w1, e2)
        score_org = BN.make_scores()

        # Consumers Net
        w1 = row['Average positiveness']
        e1 = row['Mensions']
        w2 = row['aws1']
        e2 = row['Positiveness']
        w3 = row['aws3']
        e3 = row['aws2']
        dt = row['aws4']

        BN = CustomerBayesNet()
        BN.bake_network(e1, w1, e2, w2, e3, w3, dt)
        score_customers = BN.make_scores()

        # Make dict
        data_scores[row['Url']] = [score_product, score_tech, score_org, score_customers, row['target']]

    result_df = pd.DataFrame.from_dict(data_scores, orient='index', columns=['product', 'tech', 'org', 'customers', 'target'])

    return result_df


if __name__ == '__main__':
    df = make_net_scores()
    df.to_excel(r'C:\Users\maxim\OneDrive\Desktop\folder\diplom\data\parsing\companies_scores.xlsx')
