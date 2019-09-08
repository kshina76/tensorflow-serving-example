import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


X_FEATURE = 'x'
OUTPUTS = 'outputs'
SERVING_HOST = 'localhost'
SERVING_PORT = 8500


def main():
    #mnist = input_data.read_data_sets('tmp/MNIST_data/', one_hot=True)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'predict'
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    #inputsはsignature_def_mapで指定した変数名(サーバ側で指定した変数名)
    request.inputs[X_FEATURE].CopyFrom(tf.make_tensor_proto(values=np.array([[1,1,1,1,1]],dtype=np.float32)))

    channel = implementations.insecure_channel(SERVING_HOST, SERVING_PORT)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    stub.Predict.future(request, 1)  #ここいるか？？
    result_future = stub.Predict.future(request, 1)  #「1」は、出力の個数か次元か？
    result = result_future.result()
    print(result.outputs[OUTPUTS].float_val)
    #print('answer: {}, prediction: {}'.format(np.argmax(mnist.test.labels[0]), np.argmax(result.outputs[OUTPUTS].float_val)))


if __name__ == '__main__':
    main()
