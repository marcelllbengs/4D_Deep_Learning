
import model_lib 

def getModel(mdlParams):
  """Returns a function for a model
  Args:
    mdlParams: dictionary, contains configuration
    is_training: bool, indicates whether training is active
  Returns:
    model: A function that builds the desired model
  Raises:
    ValueError: If model name is not recognized.
  """
  if mdlParams['model_type'] == 'MixedCNN_convGRU_CNN': 
      model = model_lib.MixedCNN_convGRU_CNN(mdlParams)
  elif mdlParams['model_type'] == 'nPath_NetnD': 
      model= model_lib.nPath_DenseNetnD(mdlParams)
  else:
    print('No model selected',  mdlParams['model_type'])

  return model


