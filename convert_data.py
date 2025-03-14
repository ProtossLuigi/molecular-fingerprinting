import pandas as pd
import os

os.makedirs('data/feather', exist_ok=True)

print('Loading DMSO')
df = pd.read_excel('data/xlsx/DMSO_measurements.xlsx')
print('Converting DMSO')
df.to_feather('data/feather/DMSO_measurements.feather')

print('Loading QC')
df = pd.read_excel('data/xlsx/QC_measurements.xlsx')
print('Converting QC')
df.to_feather('data/feather/QC_measurements.feather')

print('Loading FRS')
df = pd.read_excel('data/xlsx/FRS_simulations_with_sex_and_lung_cancer.xlsx')
print('Converting FRS')
df.to_feather('data/feather/FRS_simulations_with_sex_and_lung_cancer.feather')
