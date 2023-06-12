import model_lib

def getModel(mdlParams):
  """
  Returns a function for a model
  
  Args:
    mdlParams: dictionary, contains configuration
      is_training: bool, indicates whether training is active
      
  Returns:
    model: A function that builds the desired model
      
  Raises:
    ValueError: If model name is not recognized.
  """
  
  # Check the model type specified in mdlParams
  if mdlParams['model_type'] == 'MixedCNN_convGRU_CNN': 
      # Instantiate the MixedCNN_convGRU_CNN model from model_lib
      model = model_lib.MixedCNN_convGRU_CNN(mdlParams)
  elif mdlParams['model_type'] == 'nPath_NetnD': 
      # Instantiate the nPath_DenseNetnD model from model_lib
      model = model_lib.nPath_DenseNetnD(mdlParams)
  else:
    # If the model type is not recognized, print a message
    print('No model selected:', mdlParams['model_type'])
  
  # Return the selected model
  return model
