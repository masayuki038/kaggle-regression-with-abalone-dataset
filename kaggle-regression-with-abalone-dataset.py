# <a href="https://colab.research.google.com/github/masayuki038/kaggle-regression-with-abalone-dataset/blob/main/kaggle-regression-with-abalone-dataset.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# +

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'playground-series-s4e4:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F72489%2F8096274%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240404%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240404T150044Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D6a9aa0bc7fd9ff0364a494a1a6e23006d17d9bf94541bc17d376166d87b9618ae0c55dbd5bccc800798f300dbd20116212d8e6aa2b1c3425c0080dbc60e6afd6efa4dd07ef72bb7bae64e45bc4353a4e8411af30e6c0462a63126fbc1e227dad9389ba9c0bdbe8e9c88332c2a89853574bb637387857b4d60cc9d4aa2bed5bb0314ca5e8282f2964a798c135a4b2bd3d6b6f1fc9b102e4309650c82d45694080d34331ac8f6a80c565ca3651570c1c10d6010c55c78d23f4d520cba968a98242d366dd528e9a0b43e85ed70c70d57728fe151e276482ebf2ecb959e23a35b53b91d55330aebbe1783203e0784b11afbc3814a821647ac0fbf84a82bbe0c6e3ad'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

# !umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

# -

# <div style="border-radius: 15px;
#             border: 2px solid red;
#             color:white;
#             font-family: Helvetica;
#             text-align: center;
#             padding: 20px;
#             width: 800px;
#             text-align: center;
#             background-color: #FF7F7F;
#             font-size: 44px;
#             text-shadow: 2px 2px 4px #000;">
#     Short & Easy Notebook </br>
#     to get </br>
#     Started
# </div>
#

# +
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -

# <div style="border-radius: 15px;
#             border: 2px solid blue;
#             color:white;
#             font-family: Helvetica;
#             text-align: center;
#             padding: 20px;
#             width: 800px;
#             text-align: center;
#             background-color: #ADD8E6;
#             font-size: 44px;
#             text-shadow: 2px 2px 4px #000;">
#     Load Data
# </div>
#

train_data = pd.read_csv('/kaggle/input/playground-series-s4e4/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')

train_data.head().style.background_gradient()

test_data.head().style.background_gradient()

# <div style="border-radius: 15px;
#             border: 2px solid blue;
#             color:white;
#             font-family: Helvetica;
#             text-align: center;
#             padding: 20px;
#             width: 800px;
#             text-align: center;
#             background-color: #ADD8E6;
#             font-size: 44px;
#             text-shadow: 2px 2px 4px #000;">
#     Remove Duplicates & replace null values
# </div>

train_data.nunique()

train_data.isnull().sum().sum()

test_data.isnull().sum().sum()

# +
cols = [col for col in train_data.columns]

for col in cols:
    print(f"dtype of {col}: {train_data[col].dtype}")
# -

# <div style="border-radius: 15px;
#             border: 2px solid blue;
#             color:white;
#             font-family: Helvetica;
#             text-align: center;
#             padding: 20px;
#             width: 800px;
#             text-align: center;
#             background-color: #ADD8E6;
#             font-size: 44px;
#             text-shadow: 2px 2px 4px #000;">
#     Preprocessing
# </div>

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocess = ColumnTransformer([
    ('One Hot', OneHotEncoder(handle_unknown = 'ignore'), make_column_selector(dtype_include = 'object')),
    ('Scale', StandardScaler(), make_column_selector(dtype_include = 'float64'))
])

# +
Y_train = train_data['Rings']
X_train = train_data.drop(['id', 'Rings'], axis = 1)

X_test = test_data.drop(['id'], axis = 1)

train_cols = np.array(X_train.columns)
test_cols = np.array(X_test.columns)

# +
preprocess.fit(pd.concat([X_train, X_test]))

X_train = pd.DataFrame(preprocess.transform(X_train))
X_test = pd.DataFrame(preprocess.transform(X_test))

# +
cat_one_hot_cols = np.array(['Sex_F', 'Sex_I', 'Sex_M'])

train_cols = np.concatenate((cat_one_hot_cols, train_cols[1:]), axis = 0)
test_cols = np.concatenate((cat_one_hot_cols, test_cols[1:]), axis = 0)

X_train.columns = train_cols
X_test.columns = test_cols
# -

X_train.head().style.background_gradient()

X_test.head().style.background_gradient()

import seaborn as sns
import matplotlib.pyplot as plt

# +
corr = X_train.corr()

plt.figure(figsize = (10, 10), dpi = 200)
sns.heatmap(corr, cmap = 'YlGnBu', fmt = '.2f', annot = True)
plt.title('Correlation Heatmap')
plt.show()
# -

# <div style="border-radius: 15px;
#             border: 2px solid blue;
#             color:white;
#             font-family: Helvetica;
#             text-align: center;
#             padding: 20px;
#             width: 800px;
#             text-align: center;
#             background-color: #ADD8E6;
#             font-size: 44px;
#             text-shadow: 2px 2px 4px #000;">
#     Model training & predictions
# </div>

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

X_train1, X_test1, Y_train1, Y_test = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

# +
params = {
    'n_estimators': 150,
    'n_jobs': -1,
    'lambda_l1': 0.02,
    'lambda_l2': 0.06,
    'metric': 'rmse',
    'verbose': -1,
    'random_state': 42
}

model = lgb.LGBMRegressor(**params)

# +
model.fit(X_train1, Y_train1)

lgb.plot_importance(model, importance_type="gain", figsize=(12,8), max_num_features=12,
                    title="LightGBM Feature Importance (Gain)")

plt.show()

# +
preds = model.predict(X_test1)

print(f'RMSLE Score: {np.sqrt(mean_squared_error(np.log1p(preds), np.log1p(Y_test)))}')
# -

preds = model.predict(X_test)

submission = pd.DataFrame({'id': test_data.id, 'Rings': preds})
submission.to_csv('submission.csv', index = False)


