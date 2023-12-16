import random
import pandas as pd 
import numpy as np

id_1 = 7432
id_2 = 7621
id_3 = 7577
random_seed = id_1 + id_2 + id_3
random.seed(random_seed)
data_path = "Data.csv"
output_path = "tahkeer_data.csv"

all_data = pd.read_csv(data_path)
all_columns = all_data.columns.tolist()

target_column = 'smoking'
id_column = 'id'

all_columns.remove(target_column)
all_columns.remove(id_column)

selected_columns = random.sample(all_columns, 15)

print(selected_columns)  # MUST BE PRINTED
selected_columns = np.append(selected_columns, target_column)
sample_df = all_data[selected_columns].copy()
sample_df.to_csv(output_path)  # From HERE YOU CAN SPLIT FOR TRAIN ,VALID AND TEST
