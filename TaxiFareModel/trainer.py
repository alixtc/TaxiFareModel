# imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from TaxiFareModel.data import get_data, clean_data, AWS_BUCKET_PATH
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse

from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.MLFLOW_URI = "https://mlflow.lewagon.co/"
        self.experiment_name = '[FR][Marseille][alixtc] TaxiFareModel - 1.0'

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        
        # Pipelines to process Distance and Time columns
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        # combine two previous pipes on specified columns
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pass
    
        # Add the model of your choice to the pipeline
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe
        return self


    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        
        self.mlflow_log_metric("rmse", rmse)
        self.mlflow_log_param("model", self.pipeline[-1])
        return rmse
    
    # ------------- ML Flow --------------
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)



if __name__ == "__main__":    
    # store the data in a DataFrame
    df = get_data(nrows=1000)
    df = clean_data(df, test=False)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # build pipeline
    my_trainer = Trainer(X_train, y_train)

    # train the pipeline
    my_trainer.set_pipeline()
    my_trainer.run()

    # evaluate the pipeline
    rmse = my_trainer.evaluate(X_val, y_val)
    print(rmse)
