from src.data import SnakeData
import numpy as np
import warnings

with warnings.catch_warnings():  
	warnings.filterwarnings("ignore",category=FutureWarning)
	from keras.models import Sequential
	from keras.layers import Dense

if __name__ == "__main__":
	data = SnakeData()
	data.generate(1000, 1000)
	data.save()

	model = Sequential()
	model.add(Dense(units=9,input_dim=7))
	model.add(Dense(units=15, activation='relu'))
	model.add(Dense(output_dim=3,  activation='softmax'))

	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	model.fit(np.array(data.train_x).reshape(-1,7), np.array(data.train_y).reshape(-1,3), batch_size=256,epochs=10)

	model.save('data/model.h5')