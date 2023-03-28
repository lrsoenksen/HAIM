###########################################################################################################
#                                       Length of Stay Modelling
###########################################################################################################
# 
# Licensed under the Apache License, Version 2.0**
# You may not use this file except in compliance with the License. You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is 
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
# implied. See the License for the specific language governing permissions and limitations under the License.

#-> Authors: 
#      Luis R Soenksen (<soenksen@mit.edu>),
#      Yu Ma (<midsumer@mit.edu>),
#      Cynthia Zeng (<czeng12@mit.edu>),
#      Leonard David Jean Boussioux (<leobix@mit.edu>),
#      Kimberly M Villalobos Carballo (<kimvc@mit.edu>),
#      Liangyuan Na (<lyna@mit.edu>),
#      Holly Mika Wiberg (<hwiberg@mit.edu>),
#      Michael Lingzhi Li (<mlli@mit.edu>),
#      Ignacio Fuentes (<ifuentes@mit.edu>),
#      Dimitris J Bertsimas (<dbertsim@mit.edu>),
# -> Last Update: Dec 30th, 2021

import pandas as pd
from prediction_util import *

# Supply the embedding file 
fname = fname
df = pd.read_csv(fname)
df_alive_small48 = df[((df['img_length_of_stay'] < 48) & (df['death_status'] == 0))]
df_alive_big48 = df[((df['img_length_of_stay'] >= 48) & (df['death_status'] == 0))]
df_death = df[(df['death_status'] == 1)]

df_alive_small48['y'] = 1
df_alive_big48['y'] = 0
df_death['y'] = 0
df = pd.concat([df_alive_small48, df_alive_big48, df_death], axis = 0)

df = df.drop(['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay', 
        'death_status'], axis = 1)

data_type_dict = get_data_dict(df)
all_types_experiment = get_all_dtypes()
        
# Index of which we run the experiment on, this is for the sake of parallelize all experiments
ind = ind
data_type, model = all_types_experiment[ind]
run_models(data_fusion(data_type), data_type, model)


