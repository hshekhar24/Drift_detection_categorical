from google.protobuf.json_format import MessageToJson
import tensorflow_data_validation as tfdv
from zipfile import ZipFile 
import pandas as pd





#Extracting files and storing them in given file path
def extract_files(file_name):
  
  try:
    
    with ZipFile(file_name, 'r') as zip:
      
      zip.printdir()
      
      #zip.extractall(file_path)
      
      df_A = pd.read_csv(zip.extract("Reference_data.csv"))
      df_B = pd.read_csv(zip.extract("Drifted_data.csv"))
      
      print("done")
    
    return df_A, df_B
    
  except FileNotFoundError:
    print("File does not exist")
        
  except Exception as e:
    #print(e)
    return e
        


def catdrift(df_A, df_B):  
  
  col = df_A.columns
  
  #preprocessing
  for i in col:
    df_A[i].fillna(df_A[i].mode()[0], inplace = True)
    df_B[i].fillna(df_B[i].mode()[0], inplace = True)
  
    if df_A[i].dtypes == 'float':
      df_A[i] = df_A[i].astype('int')
    if df_B[i].dtypes == 'float':
      df_B[i] = df_B[i].astype('int')
  
    df_A[i] = df_A[i].astype('str')
    df_B[i] = df_B[i].astype('str')
  
    
  df_A_stats = tfdv.generate_statistics_from_dataframe(df_A,stats_options=tfdv.StatsOptions(),n_jobs = 1)
  df_B_stats = tfdv.generate_statistics_from_dataframe(df_B,stats_options=tfdv.StatsOptions(),n_jobs = 1)
  
  
  schema = tfdv.infer_schema(df_A_stats)
  
  
  for i in col:
      tfdv.get_feature(schema, i).drift_comparator.infinity_norm.threshold = 0.00
    
    
  try:
    
    drift_anomalies = tfdv.validate_statistics(statistics=df_B_stats, schema=schema, previous_statistics=df_A_stats)
    driftJson = MessageToJson(drift_anomalies)
    df_JSON = pd.read_json(driftJson)
    df_anomaly = df_JSON.anomalyInfo
    
  except AttributeError:
    return "No Drift Detected"
  
  
  
  anomalylist = list(df_anomaly.keys())
    
  feature_anomaly = []
  des = []

  
  
  for i in col:
      for j in anomalylist:
          if i == j:
              
              feature_anomaly.append(i)
              strb = df_anomaly[i]['description'].partition(" (up to ")[0]
              des.append(strb)

  df_datadrift = pd.DataFrame()
  df_datadrift['Feature'] = feature_anomaly
  df_datadrift['Drift Description'] = des
  
  
  return df_datadrift