import pandas as pd
import numpy as np

trn = pd.read_csv('C:\\Users\\bitcamp\\.kaggle\\competitions\\train_ver2.csv')
print(trn.shape)
print(trn.head())
'''
   fecha_dato  ncodpers ind_empleado pais_residencia sexo  age  fecha_alta  ...  ind_reca_fin_ult1 ind_tjcr_fin_ult1  ind_valo_fin_ult1 ind_viv_fin_ult1 ind_nomina_ult1 ind_nom_pens_ult1 ind_recibo_ult1
0  2015-01-28   1375586            N              ES    H   35  2015-01-12  ...                  0                 0                  0                0             0.0               0.0               0
1  2015-01-28   1050611            N              ES    V   23  2012-08-10  ...                  0                 0                  0                0             0.0               0.0               0
'''
for col in trn.columns:
    print('{}\n'.format(trn[col].head()))

print(trn.info())

num_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['int64', 'float64']]
print(trn[num_cols].describe())

'''
           ncodpers     ind_nuevo        indrel     tipodom      cod_prov  ind_actividad_cliente         renta
count  1.364731e+07  1.361958e+07  1.361958e+07  13619574.0  1.355372e+07           1.361958e+07  1.085293e+07
mean   8.349042e+05  5.956184e-02  1.178399e+00         1.0  2.657147e+01           4.578105e-01  1.342543e+05
std    4.315650e+05  2.366733e-01  4.177469e+00         0.0  1.278402e+01           4.982169e-01  2.306202e+05
min    1.588900e+04  0.000000e+00  1.000000e+00         1.0  1.000000e+00           0.000000e+00  1.202730e+03
25%    4.528130e+05  0.000000e+00  1.000000e+00         1.0  1.500000e+01           0.000000e+00  6.871098e+04
50%    9.318930e+05  0.000000e+00  1.000000e+00         1.0  2.800000e+01           0.000000e+00  1.018500e+05
75%    1.199286e+06  0.000000e+00  1.000000e+00         1.0  3.500000e+01           1.000000e+00  1.559560e+05
max    1.553689e+06  1.000000e+00  9.900000e+01         1.0  5.200000e+01           1.000000e+00  2.889440e+07
'''

cat_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['O']]
print(trn[cat_cols].describe())

'''
        fecha_dato ind_empleado pais_residencia      sexo       age  fecha_alta  antiguedad ult_fec_cli_1t  indrel_1mes tiprel_1mes   indresi    indext conyuemp canal_entrada   indfall   nomprov           segmento
count     13647309     13619575        13619575  13619505  13647309    13619575    13647309          24793   13497528.0    13497528  13619575  13619575     1808      13461183  13619575  13553718           13457941
unique          17            5             118         2       235        6756         507            223         13.0           5         2         2        2           162         2        52                  3
top     2016-05-28            N              ES         V        23  2014-07-28           0     2015-12-24          1.0           I         S         N        N           KHE         N    MADRID  02 - PARTICULARES
freq        931453     13610977        13553710   7424252    542682       57389      134335            763    7277607.0     7304875  13553711  12974839     1791       4055270  13584813   4409600            7960220
'''

for col in cat_cols:
    uniq = np.unique(trn[col].astype(str))
    print('-' * 50)
    print('# col {}, n_uniq {}, unqi {}'.format(col, len(uniq), uniq))

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
skip_cols = ['ncodpers', 'renta']
'''
for col in trn.columns:
    # 출력에 너무 시간이 많이 걸리는 두 변수는 skip한다
    if col in skip_cols:
        continue
    # 보기 편하게 영역 구분과 변수명을 출력한다.
    print('-' * 50)
    print('col : ', col)

    # 그래프 크기를 (figsize) 설정한다.
    f, ax = plt.subplots(figsize=(20, 15))
    # seaborn을 사용한 막대 그래프를 생성
    sns.countplot(x=col, data=trn, alpha=0.5)
    # show() 함수를 통해 시각화한다.
    plt.show()
'''
# 날짜 데이터를 기준으로 분석하기 위하여, 날짜 데이터 별도로 추출한다.
months = trn['fecha_dato'].unique().tolist()
# 제품 변수 24개 추출
label_cols = trn.columns[24:].tolist()

label_over_time = []
for i in range(len(label_cols)):
    # 매월 각 제품의 총합을 groupby(..).agg('sum')으로 계산하여, label_sum에 저장한다.
    label_sum = trn.groupby(['fecha_dato'])[label_cols[i]].agg('sum')
    label_over_time.append(label_sum.tolist())

label_sum_over_time = []
for i in range(len(label_cols)):
    # 누적 막대 그래프를 시각화하기 위하여, n번째 제품의 총합을 1 ~ n번째 제품의 총합으로 만든다.
    label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0))

# 시각화를 위하여 색깔을 지정한다.
color_list = ['#F5B7B1', '#D2B4DE', '#AED6F1', '#A2D9CE', '#ABEBC6', '#F9E79F', '#F5CBA7', '#CCD1D1']

# 그림 크기를 사전에 정의
f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    # 24개 제품에 대하여 Histogram을 그린다
    # x축에는 월 데이터, y축에는 누적 총합, 색깔은 8개 번갈아 가며 사용하며, 그림의 aplha값은 0.7로 지정
    sns.barplot(x=months, y=label_sum_over_time[i], color = color_list[i%8], alpha=0.7)

# 우측 상단에 Legend를 추가한다.
plt.legend([plt.Rectangle((0, 0),1, 1, fc=color_list[1%8], edgecolor='none')for i in range(len(label_cols))], label_cols, loc=1, ncol=2, prop={'size':16})
plt.show()
# label_sum_over_time의 값을 퍼센트 단위로 변환한다. 월마다 최댓값으로 나누고 100을 곱해준다.
label_sum_percent = (label_sum_over_time / (1.*np.asarray(label_sum_over_time).max(axis=0))) * 100

# 앞선 코드와 동일한, 시각화 실행코드이다.
f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_percent[i], color = color_list[i%8], alpha=0.7)
plt.legend([plt.Rectangle((0, 0),1, 1, fc=color_list[1%8], edgecolor='none')for i in range(len(label_cols))], label_cols, loc=1, ncol=2, prop={'size':16})
plt.show()
