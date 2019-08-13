#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout  
from keras import regularizers

model = Sequential() # Sequential = 순차적인 모델

# input_dim : 컬럼의 갯수 (행과는 상관없이 열만 맞으면 적합)
# input_shape=(1, ) - ??행 1열 [데이터 추가 삭제가 용이하다.]
# model.add(Dense(5, input_dim = 1, activation = 'relu'))
#regularizers (일반화) l1규제 적용 => kernel_regularizer=regularizers.l1(0.01) [overfit 을 막기위한 작업] <l2는 제곱해서 규제, l1은 절대값으로 규제>
model.add(Dense(16, input_shape = (3, ), activation = 'relu', kernel_regularizer=regularizers.l2(0.01))) # input과 output값 변경
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1)) 

# model.summary() # param = line 갯수 (bias가 하나의 노드)
######################### model save 방법 ##############################
model.save('savetest01.h5')
print("save clear")