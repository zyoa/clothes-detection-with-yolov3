import torch
import torch.optim as optim
import tqdm

from model import YOLOv3
import option
import utils.dataloader


def main():
    args = option.get_args()
    
    if args.mode == 'train':
        train(args)
    
    elif args.mode == 'test':
        test(args)
    
    elif args.mode == 'darknet':
        utils.pretrain_imagenet.train(args)
        
    else:
        print('wrong mode!')


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = YOLOv3()
    checkpoint = torch.load(args.yolo)
    model.load_state_dict(checkpoint['state_dict'])
    
    train_loader = utils.dataloader.get_train_loader()
    valid_loader = utils.dataloader.get_valid_loader()
        
    optimizer = optim.Adam(model.parameters())
    if checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    if checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    start_epochs = checkpoint['epochs']
    end_epochs = start_epochs + args.epochs
    
    for epoch in range(start_epochs, end_epochs):
        model.train()
        train_loss = 0
        train_style_acc = 0
        train_detail_acc = 0
        
        for images, targets in tqdm.tqdm(train_loader):
            images.to(device)
            targets.to(device)
            
            loss, output = model(images, targets)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss
            train_style_acc += 1
            train_detail_acc += 10
        
        
        if epoch % args.log_freq == 0:
            model.eval()
            valid_loss = 0
            valid_style_acc = 0
            valid_detail_acc = 0
            
            with torch.no_grad():
                for images, targets in tqdm.tqdm(valid_loader):
                    images.to(device)
                    targets.to(device)
                    
                    loss, output = model(images, targets)
                    
                    train_loss += loss
                    valid_style_acc += 1
                    valid_detail_acc += 10
            
            
            print(f'{train_loss} {train_style_acc} {train_style_acc}')
            print(f'{valid_loss} {valid_style_acc} {valid_detail_acc}')
            
            metric = {
                'train_style_acc': train_style_acc,
                'train_detail_acc': train_detail_acc,
                'valid_style_acc': valid_style_acc,
                'valid_detail_acc': valid_detail_acc
            }
            
            checkpoint = {
                'epochs': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'metric': metric
            }
            
            torch.save(checkpoint, f'model/yolo_{epoch}.pt')
        
        scheduler.step()



def test(args):
    model = YOLOv3()
    checkpoint = torch.load(args['yolo'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


if __name__ == '__main__':
    main()