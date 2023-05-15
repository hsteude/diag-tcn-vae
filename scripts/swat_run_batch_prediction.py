import os


def load_model(logs_dir, ):
    checkpoint = os.path.join(logs_dir, f'{os.listdir(loggs_dir)[0]}')
    ae_model = VanillaTcnAE.load_from_checkpoint(checkpoint)



if __name__ == '__main__':
    loggs_dir = './logs/VanillaTcnAE/version_6/checkpoints/'




