from models import BaselineTrainer
import yaml

if __name__ == '__main__':

    with open('./models/baseline.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    trainer = BaselineTrainer(cfg=config)
    trainer.train_baseline()
    trainer.test_baseline()