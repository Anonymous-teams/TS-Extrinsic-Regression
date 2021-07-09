import tensorflow as tf
import numpy as np
import time

from models.deep_learning.deep_learning_models import DLRegressor,plot_epochs_metric
from utils.tools import save_train_duration, save_test_duration


class MLPRegressor(DLRegressor):
    """
    This is a class implementing the MLP model for time series regression.
    The code is adapted from https://github.com/Anonymous-teams/Challenge-TeamJ/tree/master/MLP_model
    """

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=1000,
            batch_size=64,
            loss="mean_squared_error",
            metrics=None
    ):
        """
        Initialise the DL model

        Inputs:
            output_directory: path to store results/models
            input_shape: input shape for the models
            verbose: verbosity for the models
            epochs: number of epochs to train the models
            batch_size: batch size to train the models
            loss: loss function for the models
            metrics: metrics for the models
        """
        self.name = "MLP"
        super().__init__(
            output_directory=output_directory,
            input_shape=input_shape,
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics
        )

    def build_model(self, input_shape):
        """
        Build the MLP model

        Inputs:
            input_shape: input shape for the model
        """
        # input_layer = tf.keras.layers.Input(input_shape)
        # output_layer = tf.keras.layers.Dense(1, activation='linear')(gap_layer)
        # model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(input_shape*10, input_shape=(input_shape,)))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dense(input_shape*2))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        model.compile(loss=self.loss,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=self.metrics)

        return model
    
    def create_derivative_features(self,x):
        """ add derivative features of all sensors """
        # if (x==None):
        #     return None
        N = x.shape[1]
        X = np.zeros((x.shape[0],2*N),dtype=np.float32)
        X[:,:N] = x[:,:N]
        for j in range(N):
            X[1:,N+j] = np.diff(X[:,j],axis=0)
            X[0,N+j] = X[1,N+j]
        return X

    def fit(self, x_train, y_train, x_val=None, y_val=None, monitor_val=False):
        """
        Fit DL models

        Inputs:
            x_train: training data (num_examples, num_timestep, num_channels)
            y_train: training target
            x_val: validation data (num_examples, num_timestep, num_channels)
            y_val: validation target
            monitor_val: boolean indicating if model selection should be done on validation
        """
        print('[{}] Training'.format(self.name))
        

        start_time = time.perf_counter()

        if len(x_train.shape) == 3:
            x_train = x_train[:,0,:] # x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
            x_val = x_val[:,0,:] # x_val.reshape(x_val.shape[0], x_val.shape[1] * x_val.shape[2])

        x_train = self.create_derivative_features(x_train)
        x_val = self.create_derivative_features(x_val)        

        self.X_train = x_train
        self.y_train = y_train
        self.X_val = x_val
        self.y_val = y_val

        epochs = self.epochs
        batch_size = self.batch_size
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
        if (x_train.shape[0] / 10)>1024:
            mini_batch_size = 1024

        file_path = self.output_directory + self.best_model_file
        if (x_val is not None) and monitor_val:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                             factor=0.5, patience=50,
                                                             min_lr=0.0001)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                                  monitor='val_loss',
                                                                  save_best_only=True)
        else:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                             factor=0.5, patience=50,
                                                             min_lr=0.0001)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                                  monitor='loss',
                                                                  save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

        # train the model
        if x_val is not None:
            self.hist = self.model.fit(self.X_train, y_train,
                                       validation_data=(x_val, y_val),
                                       verbose=self.verbose,
                                       epochs=epochs,
                                       batch_size=mini_batch_size,
                                       callbacks=self.callbacks)
        else:
            self.hist = self.model.fit(self.X_train, y_train,
                                       verbose=self.verbose,
                                       epochs=epochs,
                                       batch_size=mini_batch_size,
                                       callbacks=self.callbacks)

        self.train_duration = time.perf_counter() - start_time

        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)

        print('[{}] Training done!, took {}s'.format(self.name, self.train_duration))

        plot_epochs_metric(self.hist,
                           self.output_directory + 'epochs_loss.png',
                           metric='loss',
                           model=self.name)
        for m in self.metrics:
            plot_epochs_metric(self.hist,
                               self.output_directory + 'epochs_{}.png'.format(m),
                               metric=m,
                               model=self.name)

    def predict(self, x):
        """
        Do prediction with DL models

        Inputs:
            x: data for prediction (num_examples, num_timestep, num_channels)
        Outputs:
            y_pred: prediction
        """
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()
        model = tf.keras.models.load_model(self.output_directory + self.best_model_file)
        
        if len(x.shape) == 3:
            x = x[:,0,:] # x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        yhat = model.predict(self.create_derivative_features(x))

        tf.keras.backend.clear_session()
        test_duration = time.perf_counter() - start_time

        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)

        print('[{}] Prediction done!'.format(self.name))

        return yhat
