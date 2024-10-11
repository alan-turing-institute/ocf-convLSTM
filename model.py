from cloudcasting.models import AbstractModel
from ocf_convLSTM.first_model import FirstModel
import numpy as np
import torch

# We define a new class that inherits from AbstractModel
class ConvLSTM(AbstractModel):
    """ConvLSTM model class"""

    def __init__(self, history_steps: int, state_dict_path: str) -> None:
        # All models must include `history_steps` as a parameter. This is the number of previous
        # frames that the model uses to makes its predictions. This should not be more than 25, i.e.
        # 6 hours (inclusive of end points) of 15 minutely data.
        # The history_steps parameter should be specified in `validate_config.yml`, along with
        # any other parameters (replace `example_parameter` with as many other parameters as you need to initialize your model, and also add them to `validate_config.yml` under `model: params`)
        super().__init__(history_steps)


        ###### YOUR CODE HERE ######
        # Here you can add any other parameters that you need to initialize your model
        # You might load your trained ML model or set up an optical flow method here.
        # You can also access any code from src/ocf_convLSTM, e.g.
        self.model = FirstModel()
        self.model.load_state_dict(torch.load(state_dict_path, weights_only=True))
        self.model.eval()
        ############################


    def forward(self, X):
        # This is where you will make predictions with your model
        # The input X is a numpy array with shape (batch_size, channels, time, height, width)

        ###### YOUR CODE HERE ######
        frames = []
        X_new = X.copy()
        
        for i in range(0,12):
            y_hat = self.model(torch.Tensor(X_new)).detach().cpu().numpy()
            y_hat = y_hat[:, :, np.newaxis, ...]
            frames.append(y_hat)
            X_new = X_new[:,:,1:]
            X_new = np.concatenate([X_new,y_hat], axis=2)
        
        y_hat = np.concatenate(frames, axis=2)

        return y_hat


    def hyperparameters_dict(self):

        # This function should return a dictionary of hyperparameters for the model
        # This is just for your own reference and will be saved with the model scores to wandb

        ###### YOUR CODE HERE ######
        params_dict =  {
            "epoch": 2,
            "learning_rate": 0.001
        }

        return params_dict
      
