from utils.data_aug import create_data_aug_layer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from tensorflow.keras.applications import resnet50





def create_model(
    weights: str = "imagenet",
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.0,
    data_aug_layer: dict = None,
    classes: int = None,
):
    """
    Creates and loads the Resnet50 model we will use for our experiments.
    Depending on the `weights` parameter, this function will return one of
    two possible keras models:
        1. weights='imagenet': Returns a model ready for performing finetuning
                               on your custom dataset using imagenet weights
                               as starting point.
        2. weights!='imagenet': Then `weights` must be a valid path to a
                                pre-trained model on our custom dataset.
                                This function will return a model that can
                                be used to get predictions on our custom task.

    See an extensive tutorial about finetuning with Keras here:
    https://www.tensorflow.org/tutorials/images/transfer_learning.

    Parameters
    ----------
    weights : str
        One of None (random initialization),
        'imagenet' (pre-training on ImageNet), or the path to the
        weights file to be loaded.

    input_shape	: tuple
        Model input image shape as (height, width, channels).
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the input shape defined and we shouldn't change it.
        Input image size cannot be no smaller than 32. E.g. (224, 224, 3)
        would be one valid value.

    dropout_rate : float
        Value used for Dropout layer to randomly set input units
        to 0 with a frequency of `dropout_rate` at each step during training
        time, which helps prevent overfitting.
        Only needed when weights='imagenet'.

    data_aug_layer : dict
        Configuration from experiment YAML file used to setup the data
        augmentation process during finetuning.
        Only needed when weights='imagenet'.

    classes : int
        Model output classes.
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the output classes number defined and we shouldn't change
        it.

    Returns
    -------
    model : keras.Model
        Loaded model either ready for performing finetuning or to start doing
        predictions.
    """

    if weights == "imagenet":

        #   1. layer dtype must be tensorflow.float32
    
        input = keras.Input(dtype= tf.float32, shape= input_shape)

        # Create the data augmentation layers and add to the model next
        # to the input layer

        if data_aug_layer is not None:
            x = create_data_aug_layer(data_aug_layer)(input)
            x = resnet50.preprocess_input(x)
        else:
            x = resnet50.preprocess_input(input)
      
         
        base_model = resnet50.ResNet50(
            include_top = False,
            weights=weights, 
            pooling='avg',
            )
        
        x = base_model(x) 

        # Add a single dropout layer for regularization.
        x = layers.Dropout(dropout_rate)(x)

        # Add the classification layer here, use keras.
        outputs = layers.Dense(
            units=classes,
            activation="softmax",
            kernel_regularizer="l2",
        )(x)
        

        # Create a new model       
       
        model = keras.Model(input, outputs)
    else:
        # For this particular case I want to load the already defined and
        # finetuned model.
        
        model =  keras.models.load_model(weights)
        

    return model
 