import pandas as pd
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, Adagrad, RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import mnist
import keras.backend as K
from keras.backend import tensorflow_backend as backend

X_FEATURE = 'x'
OUTPUTS = 'outputs'
MODEL_DIR = 'tmp/predict_price/'

os.environ['PYTHONHASHSEED'] = '0'
tf.set_random_seed(42)
np.random.seed(42)
random.seed(42)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def DNN_model():
	inputs = Input(shape=(5,))
	x = Dense(256,activation='relu')(inputs)
	x = Dense(128,activation='relu')(inputs)
	prediction = Dense(1,activation='linear')(x)
	model = Model(input=inputs, output=prediction)

	optimizer = Adam()
	model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])
	model.summary()

	return model

def save_keras_model(model):
        signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={X_FEATURE: tf.saved_model.utils.build_tensor_info(model.inputs[0])},
                outputs={OUTPUTS: tf.saved_model.utils.build_tensor_info(model.outputs[0])},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        export_dir = os.path.join(MODEL_DIR, str(int(time.time())))
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)
        with K.get_session() as sess:
                builder.add_meta_graph_and_variables(
                        sess=sess,
                        tags=[tf.saved_model.tag_constants.SERVING],
                        signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
                )
                builder.save()
        return export_dir


K.set_learning_phase(0)

#get data from csv
feature = ['open','high','low','volume']
df = pd.read_csv('./EUR_USD_H1.csv')

#make target
copy_df = df.copy()
target_s = copy_df.close.shift(-1)
target_s = target_s.drop(len(target_s)-1)
print(target_s)

#delete comp and time columns
del df['comp']
del df['time']

#make features
features = df.drop(df.shape[0]-1)
print(features.tail(5))

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(features, target_s, test_size=0.2)

#training
model = DNN_model()
history = model.fit(X_train, y_train, batch_size=50, epochs=5, validation_split=0.2)

#validation
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#save model
export_dir = save_keras_model(model)



#prediction
'''
X_test = X_test.values
print(X_test)
print(type(X_test))
a = np.array(
       [[1,1,1,1,1]]
)
print(type(a))

predicted = model.predict(a)
print(predicted)
'''

#セッションを試してみた
'''
sess = K.get_session()
kgraph = sess.graph
print(kgraph.as_graph_def())
'''



