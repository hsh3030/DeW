import pandas as pd
import numpy as np

# 훈련 데이터 읽기
trn = pd.read_csv('C:\\Users\\bitcamp\\.kaggle\\competitions\\train_ver2.csv')

# 제품 변수를 prods에 list형태로 저장한다.
prods = trn.columns[24:].tolist()

# 날짜를 숫자로 변환하는 함수이다. 2015-01-28은 1, 2016-06-28은 18로 변환된다.
def data_to_int():
    Y, M, D = [int(a) for a in str_data.strip().split("-")]
    int_data = (int(Y) - 2015) * 12 + int(M)
    return int_date

# 날짜를 숫자로 변환하여 int_date에 저장한다.
trn['int_data'] = trn['fecha_dato'].map(data_to_int).astype(np.int8)

# 데이터를 복사하고, int_date 날짜에 1을 더하여 lag를 생성한다. 변수명에 _prev를 추가한다.
trn_lag = trn.copy()
trn_lag['int_data'] += 1
trn_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_data']
else col for col in trn.columns]

# 원본 데이터와 lag데이터를 ncodper와 int_date 기준으로 합친다.
# lag 데이터의 int_date는 1 밀려 있기 때문에, 저번 달의 제품 정보가 삽입된다.
df_trn = trn.merge(trn_lag, on=['ncodpers', 'int_data'], how='left')

# 메모리 효율을 위해 불필요한 변수를 메모리에서 제거한다.
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0, inplace=True)

# 원본 데이터에서의 제품 보유 여부 - lag데이터에서의 제품 보유 여부를 비교하여 신규 구매 변수 padd를 구한다.
for prod in prods:
    padd = prod + '_add'
    prev = prod + '_prev'
    df_trn[padd] = ((df_trn[prod] == 1) & (df_trn[prev] == 0)).astype(np.int8)

# 신규 구매 변수만을 추출하여 labels에 저장한다.
add_cols = [prod + '_add' for prod in prods]
labels = df_trn[add_cols].copy()
labels.columns = prods
labels.to_csv('C:\\Users\\bitcamp\\.kaggle\\competitions\\labels.csv', index=False)
