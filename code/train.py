import os
import wandb
import argparse
from utils import *
from model import HFSNet
from trainer import trainer
from datetime import datetime
from torch.utils.data import DataLoader
from moled_dataset import moled_dataset

def main(args):
    setup_seed(args.seed)
    source_dataset = moled_dataset(data_path=args.source_data_path, name='source')
    target_dataset = moled_dataset(data_path=args.target_data_path, name='target')
    
    source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    targer_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    student_net = HFSNet()
    teacher_net = HFSNet()
    
    Trainer = trainer(source_loader=source_dataloader, target_loader=targer_dataloader, student_model=student_net, teacher_model=teacher_net, args=args)
    Trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--save_frequency', default=10, type=int)
    parser.add_argument('--source_data_path', default='Your Source Data Path', type=str)
    parser.add_argument('--target_data_path', default='Your Target Data Path', type=str)
    parser.add_argument('--save_path', default='Your Save Path', type=str)
    args = parser.parse_args()

    os.environ["WANDB_API_KEY"] = "xxx"
    wandb.login()
    nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project='FPS', config=args.__dict__, name='stroke_patients' + nowtime, save_code=False)

    main(args)