from stable_baselines3.common.callbacks import BaseCallback
import os

class SaveCallback(BaseCallback):
    def __init__(self, save_path, save_freq):
        super(SaveCallback, self).__init__()
        self.save_path = save_path
        self.save_freq = save_freq
        # Create folder if needed
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        # clear the folder
        for file in os.listdir(self.save_path):
            file_path = os.path.join(self.save_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    
    def save(self):
            model_save_name = f"model_{self.num_timesteps}.zip"
            model_save_path = os.path.join(self.save_path, model_save_name)
            self.model.save(model_save_path)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.save()
        return True