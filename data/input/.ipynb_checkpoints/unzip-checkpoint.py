import zipfile

with zipfile.ZipFile('/datadrive/workspace/s-saito/kaggle_pipeline_m5_accuracy/data/input/m5-forecasting-accuracy.zip') as existing_zip:
    existing_zip.extractall('/datadrive/workspace/s-saito/kaggle_pipeline_m5_accuracy/data/input')