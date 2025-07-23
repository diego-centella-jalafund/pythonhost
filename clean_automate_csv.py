import pandas as pd
import re
import sys

if len(sys.argv) != 2:
    sys.exit(1)

csv_file = sys.argv[1]
output_file = 'fileTest_cleaned.csv'

df = pd.read_csv(csv_file, header=None, skip_blank_lines=True)
df = df.astype(str)

metadata_rows = df[df[0].str.contains('INFORME|INFORMACIÓN|Fecha:|TABLA|PARÁMETRO|Requisitos', na=False, case=False)].index
df_clean = df.drop(metadata_rows).reset_index(drop=True)
data_start = df_clean.index[df_clean[0].str.contains('pH a 20°C', na=False, case=False)].min()
data_end = df_clean.index[df_clean[0].str.contains('TRAM', na=False, case=False)].max() + 2
if pd.isna(data_start):
    data_start = 0
if pd.isna(data_end):
    data_end = len(df_clean)

df_data = df_clean.iloc[data_start:data_end].reset_index(drop=True)

num_columns = df_data.shape[1]

df_data = df_data.dropna(how='all', axis=1)
num_columns_adjusted = df_data.shape[1]

headers = ['empty', 'parameter', 'unit', 'evening', 'early_morning', 'gmp2', 'range', 'method']
if num_columns_adjusted != len(headers):
    raise ValueError(f"Expected {len(headers)} columns, but found {num_columns_adjusted} columns")

df_data.columns = headers
df_data = df_data.drop(columns=['empty', 'method'])

param_map = {
    'pH a 20°C': ['ph_20c_evening', 'ph_20c_early_morning', 'ph_20c_gmp2'],
    'Temperatura pH': ['evening_temperature', 'early_morning_temperature', 'gmp2_temperature'],
    'Acidez titulable': ['titratable_acidity_evening', 'titratable_acidity_early_morning', 'titratable_acidity_gmp2'],
    'Densidad a 20°C': ['density_20c_evening', 'density_20c_early_morning', 'density_20c_gmp2'],
    'Materia Grasa': ['fat_content_evening', 'fat_content_early_morning', 'fat_content_gmp2'],
    'Sólidos no Grasos': ['non_fat_solids_evening', 'non_fat_solids_early_morning', 'non_fat_solids_gmp2'],
    'Alcoholimetría': ['alcohol_test_evening', 'alcohol_test_early_morning', 'alcohol_test_gmp2'],
    'TRAM (Tiempo de reducción de azul de metileno)': ['tram_evening', 'tram_early_morning', 'tram_gmp2']
}

general_info = df[df.apply(lambda row: row.str.contains('fecha:|n° muestra|hora de muestreo|temp. muestreo', case=False, na=False).any(), axis=1)]

date_row = general_info[general_info.apply(lambda row: row.str.contains('fecha:', case=False, na=False).any(), axis=1)]
if not date_row.empty:
    date_col = date_row.columns[date_row.iloc[0].str.contains('fecha:', case=False, na=False)][0]
    date_value = date_row.iloc[0][date_col]
    if ':' in date_value:
        date_parts = [re.sub(r'[^0-9]', '', part).zfill(2) for part in date_value.split(':')[1].strip().split('/')]
        if len(date_parts) == 3:
            day, month, year = date_parts
            year = f"20{year.zfill(2)}" if len(year) == 2 else year
            date = f"{year}-{month}-{day}"
        else:
            date = 'null'
    else:
        date = 'null'
else:
    date = 'null'
analysis_date = date

sample_row = general_info[general_info.apply(lambda row: row.str.contains('n° muestra', case=False, na=False).any(), axis=1)]
if not sample_row.empty:
    sample_col = sample_row.columns[sample_row.iloc[0].str.contains('n° muestra', case=False, na=False)][0]
    sample_numbers = [sample_row.iloc[0][col].strip() for col in [3, 4, 5] if pd.notna(sample_row.iloc[0][col]) and sample_row.iloc[0][col] != 'nan']
    if len(sample_numbers) != 3:
        sample_numbers = ['null', 'null', 'null']  
else:
    sample_numbers = ['null', 'null', 'null']

time_row = general_info[general_info.apply(lambda row: row.str.contains('hora de muestreo', case=False, na=False).any(), axis=1)]
if not time_row.empty:
    time_col = time_row.columns[time_row.iloc[0].str.contains('hora de muestreo', case=False, na=False)][0]
    sampling_times = [time_row.iloc[0][col].strip() for col in [3, 4, 5] if pd.notna(time_row.iloc[0][col]) and time_row.iloc[0][col] != 'nan']
    if len(sampling_times) != 3:
        sampling_times = ['', '', '']  
else:
    sampling_times = ['', '', '']

temp_row = general_info[general_info.apply(lambda row: row.str.contains('temp. muestreo', case=False, na=False).any(), axis=1)]
if not temp_row.empty:
    temp_col = temp_row.columns[temp_row.iloc[0].str.contains('temp. muestreo', case=False, na=False)][0]
    sampling_temps = [temp_row.iloc[0][col].replace(',', '.') for col in [3, 4, 5] if pd.notna(temp_row.iloc[0][col]) and temp_row.iloc[0][col] != 'nan']
    if len(sampling_temps) != 3:
        sampling_temps = ['', '', ''] 
else:
    sampling_temps = ['', '', '']

result = pd.DataFrame({
    'date': [date],
    'analysis_date': [analysis_date],
    'evening_sample_number': [sample_numbers[0]],
    'early_morning_sample_number': [sample_numbers[1]],
    'gmp2_sample_number': [sample_numbers[2]],
    'evening_sampling_time': [sampling_times[0]],
    'early_morning_sampling_time': [sampling_times[1]],
    'gmp2_sampling_time': [sampling_times[2]],
    'evening_sampling_temperature': [float(sampling_temps[0]) if sampling_temps[0] else ''],
    'early_morning_sampling_temperature': [float(sampling_temps[1]) if sampling_temps[1] else ''],
    'gmp2_sampling_temperature': [float(sampling_temps[2]) if sampling_temps[2] else '']
})

for index, row in df_data.iterrows():
    param = row['parameter']
    if param in param_map:
        cols = param_map[param]
        if param == 'TRAM (Tiempo de reducción de azul de metileno)':
            next_row = df_data.iloc[index + 1] if index + 1 < len(df_data) else None
            if next_row is not None and pd.notna(next_row['evening']):
                evening_value = next_row['evening'].replace(',', '.')
                early_morning_value = next_row['early_morning'].replace(',', '.')
                gmp2_value = next_row['gmp2'].replace(',', '.')
            else:
                evening_value = row['evening']
                early_morning_value = row['early_morning']
                gmp2_value = row['gmp2']
        else:
            evening_value = row['evening']
            early_morning_value = row['early_morning']
            gmp2_value = row['gmp2']

        result[cols[0]] = [float(evening_value.replace(',', '.')) if isinstance(evening_value, str) and evening_value.replace(',', '').replace('.', '').isdigit() else evening_value]
        result[cols[1]] = [float(early_morning_value.replace(',', '.')) if isinstance(early_morning_value, str) and early_morning_value.replace(',', '').replace('.', '').isdigit() else early_morning_value]
        result[cols[2]] = [float(gmp2_value.replace(',', '.')) if isinstance(gmp2_value, str) and gmp2_value.replace(',', '').replace('.', '').isdigit() else gmp2_value]

result.to_csv(output_file, index=False)
