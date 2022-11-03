# Modulo para el entrenamiento de modelos RNN
# Javier Martinez

import pandas as pd
import numpy as np

from tensorflow import keras


#----
class LogMinimax:
    """
    Transformacion LogMinimax
    """

    @classmethod
    def create(cls, values):

        clase = cls()

        clase.values = values
        clase.log_values = np.log(clase.values + 1)
        clase.max = clase.log_values.max()
        clase.min = clase.log_values.min()

        return clase

    def transformacion(self):

        return (self.log_values - self.min)/(self.max - self.min)

    def inversa(self,y):

        return  np.exp( ( y*(self.max - self.min) ) + self.min ) - 1



#-----
class RNN_LSTM:

    def __init__(self,item_id,pd_rebuild):
        self.item_id = item_id
        self.pd_rebuild = pd_rebuild
    
    def create_data(self,end='2022-01-23', auto_order = 30*6, transf = LogMinimax):
        """
        Funcion para crear estructura de datos
        """
        # Data para el modelo
        pd_for_model = self.pd_rebuild.query(f"item_id=={self.item_id}")
        pd_for_model.index = pd.to_datetime(pd_for_model.fecha)

        self.rubro_id = pd_for_model.rubro_id.unique()[0]


        # Agregando transformacion
        self.transformacion = transf.create( pd_for_model.venta.to_numpy() )
        pd_for_model['t_venta'] = self.transformacion.transformacion()
        pd_for_model = pd_for_model[['t_venta']].copy().sort_index()


        self.pd_for_model = pd_for_model.copy()

        # Rango de fecha deseado
        self.date_range_predit = pd.date_range(start=pd_for_model.index.max() + pd.DateOffset(days=1),
                                        end=end,
                                        #periods=None,
                                        freq='D')

        # Paametros
        self.prediction_order = len(self.date_range_predit)
        self.auto_order = auto_order# componente autoregresiva

        self.input_width = auto_order
        label_width = auto_order
        self.shift = 1

        x_data, y_data = RNN_LSTM.data_estructura(data_pd=self.pd_for_model,auto_order=self.auto_order)

        self.x_train = x_data[:-self.prediction_order]
        self.x_vasl = x_data[-self.prediction_order:]

        self.y_train = y_data[:-self.prediction_order]
        self.y_vasl = y_data[-self.prediction_order:]

    def fit_model(self,patience=20,epochs=100):

        """
        Funcion para el entrenamiento
        """


        #-----
        # Modelo RNN LSTM
        #-----

        # Metrícas
        mae = keras.metrics.MeanAbsoluteError()
        rmse = keras.metrics.RootMeanSquaredError()

        self.model = keras.models.Sequential()

        self.model.add(keras.layers.LSTM(self.auto_order, return_sequences=False ))

        self.model.add(keras.layers.Dense(1))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=[mae,rmse]) 

        callback = keras.callbacks.EarlyStopping(
                                                    monitor="loss",
                                                    min_delta=0,
                                                    patience=patience,
                                                    verbose=0,
                                                    mode="min",
                                                    baseline=None,
                                                    restore_best_weights=False,
                                                )


        self.history = self.model.fit(x=self.x_train,
                                    y=self.y_train,
                                    epochs=epochs,
                                    batch_size=1,
                                    verbose=0,
                                    workers=2,
                                    callbacks=[callback])

    
    def validation_data(self, end_date='2022-01-23'):

        """
        Funcion para metricas y validacion
        """
        #----
        # predict
        #----
        trainPredict = self.model.predict(self.x_train, verbose=0).reshape(-1)
        testPredict = self.model.predict(self.x_vasl, verbose=0).reshape(-1)

        #----
        # Training Test
        #----
        trainind_pd = pd.DataFrame(trainPredict,
                                    index = self.pd_for_model.index[:-self.prediction_order][-len(trainPredict):],
                                    columns=['prediction']
                                    )

        trainind_pd['t_venta'] = self.y_train.reshape(-1)
        trainind_pd['type'] = 'training'

        trainind_pd['venta'] = trainind_pd['t_venta'].apply(lambda x: self.transformacion.inversa(x) if np.isnan(x)==False else np.nan )
        trainind_pd['prediction_ventan'] = trainind_pd['prediction'].apply(lambda x: self.transformacion.inversa(x) if np.isnan(x)==False else np.nan )

        # metricas
        trainind_metrics =  RNN_LSTM.metrics(trainind_pd.venta, trainind_pd.prediction_ventan)

        #----
        # validacion
        #----
        validation_pd = pd.DataFrame(testPredict,
                                    index = self.pd_for_model.index[-self.prediction_order:],
                                    columns=['prediction']
                                    )

        validation_pd['t_venta'] = self.y_vasl.reshape(-1)
        validation_pd['type'] = 'validation'

        validation_pd['venta'] = validation_pd['t_venta'].apply(lambda x: self.transformacion.inversa(x) if np.isnan(x)==False else np.nan )
        validation_pd['prediction_ventan'] = validation_pd['prediction'].apply(lambda x: self.transformacion.inversa(x) if np.isnan(x)==False else np.nan )

        # metricas
        validation_metrics =  RNN_LSTM.metrics(validation_pd.venta, validation_pd.prediction_ventan)

        #----
        # Prediccion
        #----
        for_predict = RNN_LSTM.predict_one_stap(self.model, self.pd_for_model, self.auto_order, self.prediction_order)

        # Data de test
        prediction_pd = pd.DataFrame(for_predict[-len(self.date_range_predit):],
                                    index=self.date_range_predit,
                                    columns=['prediction'])

        prediction_pd['t_venta'] = np.nan 
        prediction_pd['type'] = 'prediction'

        prediction_pd['venta'] = np.nan 
        prediction_pd['prediction_ventan'] = prediction_pd['prediction'].apply(lambda x: self.transformacion.inversa(x) if np.isnan(x)==False else np.nan )

        #----
        # Resultados
        #----
        self.pd_summary = pd.concat([trainind_pd, validation_pd, prediction_pd])\
                        .reset_index(drop=False)\
                        .rename(columns={'index':'fecha'})
        self.pd_summary['item_id'] = int(self.item_id)
        self.pd_summary['rubro_id'] = int(self.rubro_id)


        self.dict_metrics = {'item_id':[self.item_id],
                        'epocas':[len(self.history.epoch)],
                        'prediction_order':[self.prediction_order],
                        'auto_order':[self.auto_order],
                        'training_mse':[trainind_metrics['mse']],
                        'training_rmse':[trainind_metrics['rmse']],
                        'training_mae':[trainind_metrics['mae']],
                        'trainig_mape':[trainind_metrics['mape']],
                        'validation_mse':[trainind_metrics["mse"]],
                        'validation_rmse':[validation_metrics["rmse"]],
                        'validation_mae':[validation_metrics["mae"]],
                        'validation_mape':[validation_metrics['mape']],
                        }

        self.experimento_pd = pd.DataFrame.from_dict(self.dict_metrics)

    #
    @staticmethod
    def data_estructura(data_pd,auto_order):
        """
        Función para la estructura de los datos
        """
        
        x_data = []
        y_data = []

        len_data = data_pd.shape[0]
        init_range = len_data-auto_order*int(len_data/auto_order)

        for i in range(init_range, len_data-auto_order):
            to_split = data_pd[(i):(auto_order+1+i)].copy().to_numpy().astype(float)
            #print(to_split.shape)
            x_data.append( to_split[:(auto_order)].reshape(-1) )
            y_data.append(to_split[-1])

        x_data = np.array(x_data)
        x_data = np.reshape(x_data, (x_data.shape[0], 1, x_data.shape[1]))
        y_data = np.array(y_data).reshape(-1)

        return x_data, y_data

    #
    @staticmethod
    def metrics(observado,prediccion):

        """
        Cálculo de las metricas del modelo
        """
        
        from sklearn.metrics import (mean_absolute_percentage_error,mean_absolute_error,mean_squared_error,r2_score)

        return {
                'mape':100*mean_absolute_percentage_error(observado, prediccion),
                'mae':mean_absolute_error(observado, prediccion),
                'mse':mean_squared_error(observado, prediccion,squared=False),
                'rmse':mean_squared_error(observado, prediccion,squared=True)
                }

    #
    @staticmethod
    def predict_one_stap(model ,data_pd, auto_order, prediction_order):

        """
        Función para la predicción
        """
        # rango de prediccion
        date_range = pd.date_range(data_pd.index.max(), periods= prediction_order+1, freq='M')[-prediction_order:]
        date_range = pd.to_datetime([x+'-01' for x in date_range.strftime('%Y-%m')])


        for_predict = data_pd[-auto_order:].to_numpy().astype(float).reshape(-1) 
        for i in range(prediction_order):

            np_predict = np.reshape(for_predict[-auto_order:], (1, 1, auto_order))

            predictionPredict = model.predict(np_predict, verbose=0).reshape(-1)

            for_predict = np.append( for_predict, predictionPredict)
            
        return for_predict