# import argparse
# import os
# import torch
# import yaml
# from torch.utils.data import DataLoader
# from utils.logger import Logger
# from utils.misc import set_seed, prepare_device, get_config_info
# from utils.metrics import accuracy, f1_score, precision, recall
# from utils.checkpoints import save_checkpoint
# from utils.lr_scheduler import LRScheduler
# from utils.transforms import build_transforms
# from utils.dataset import build_dataset
# from models import build_model
#
# def parse_args():
#     """Set up arguments parser."""
#     parser = argparse.ArgumentParser(description='Training arguments')
#
#     # experiment information
#     parser.add_argument('experiment', type=str, help='name of the experiment')
#
#     # todo:第二部分
#     # hyperparameters
#     parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
#     parser.add_argument('--val_batch_size', type=int, default=32, help='input batch size for validation')
#     parser.add_argument('--num_workers', type=int, default=8, help='number of workers for the dataloader')
#     parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
#     parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
#     parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
#     parser.add_argument('--scheduler', type=str, default='cosine', help='name of the LR scheduler')
#
#     # training options
#     parser.add_argument('--max_epochs', type=int, default=50, help='number of epochs to train')
#     parser.add_argument('--start_epoch', type=int, default=0, help='manual epoch number (useful on restarts)')
#     parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
#     parser.add_argument('--no_val', action='store_true', help='disable validation')
#
#     # data options
#     parser.add_argument('--data_dir', type=str, default='./data', help='path to dataset directory')
#     parser.add_argument('--train_dir', type=str, default='train', help='path to training dataset directory')
#     parser.add_argument('--val_dir', type=str, default='val', help='path to validation dataset directory')
#     parser.add_argument('--test_dir', type=str, default='test', help='path to test dataset directory')
#     parser.add_argument('--resize', type=int, default=256, help='resize size')
#     parser.add_argument('--crop_size', type=int, default=224, help='crop size')
#     parser.add_argument('--data_mode', type=str, default='image', help='data type (image or spectrogram)')
#     parser.add_argument('--spectrogram_params', type=dict, default={'n_fft': 2048, 'hop_length': 512},
#                         help='parameters for spectrogram calculation')
#     parser.add_argument('--normalize', action='store_true', help='normalize input images')
#     parser.add_argument('--augment', action='store_true', help='enable data augmentation')
#
#     # model options
#     parser.add_argument('--model', type=str, default='vgg', help='name of the model')
#     parser.add_argument('--pretrained', action='store_true', help='use a pretrained model')
#     parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
#     parser.add_argument('--freeze', type=int, default=0, help='freeze n layers from the beginning')
#     parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
#     parser.add_argument('--fc_channels', type=int, default=4096, help='number of channels in the last fc layer')
#     parser.add_argument('--pool_type', type=str, default='max', help='pooling type')
#     parser.add_argument('--conv_dims', type=list, default=[64, 128, 256, 512, 512],
#                         help='list of output channels for each conv layer')
#     parser.add_argument('--use_bn', action='store_true', help='use batch normalization')
#     parser.add_argument('--norm_type', type=str, default='batch', help='type of normalization layer to
#
#     # todo:第三部分
#     # loss function and optimizer
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
#
#     # learning rate scheduler
#     lr_scheduler = LRScheduler(optimizer, args.scheduler, args.learning_rate, args.max_epochs)
#
#     # prepare data loaders
#     train_transforms = build_transforms(args, is_train=True)
#     val_transforms = build_transforms(args, is_train=False)
#
#     train_dataset = build_dataset(args, data_type='train', transforms=train_transforms)
#     val_dataset = build_dataset(args, data_type='val', transforms=val_transforms)
#
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
#
#     # prepare logger and checkpoint saver
#     logger = Logger(args.experiment, log_freq=10)
#     checkpoint_saver = lambda epoch, filename: save_checkpoint(epoch, filename, model, optimizer, lr_scheduler)
#
#     # move model and criterion to device
#     device = prepare_device(args.gpu)
#     model = model.to(device)
#     criterion = criterion.to(device)
#
#     # training loop
#     best_val_acc = 0.0
#     for epoch in range(args.start_epoch, args.max_epochs):
#         logger.epoch_start()
#         train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, lr_scheduler, epoch, device)
#         logger.train_log({'loss': train_loss, 'accuracy': train_acc})
#
#         # validate the model
#         if not args.no_val:
#             val_loss, val_acc = val_epoch(val_loader, model, criterion, device)
#             logger.val_log({'loss': val_loss, 'accuracy': val_acc})
#
#             # save the best model checkpoint
#             if val_acc > best_val_acc:
#                 best_val_acc = val_acc
#                 checkpoint_saver(epoch, os.path.join(args.experiment, 'best.pth'))
#
#         logger.epoch_end()
#         checkpoint_saver(epoch, os.path.join(args.experiment, 'last.pth'))
#
#     # todo:第四部分
#     # build transforms
#     transforms = build_transforms(resize=args.resize, crop_size=args.crop_size, data_mode=args.data_mode, normalize=args.normalize, augment=args.augment)
#
#     # build datasets
#     train_dataset = build_dataset(data_dir=os.path.join(args.data_dir, args.train_dir),
#                                   transforms=transforms,
#                                   data_mode=args.data_mode,
#                                   spectrogram_params=args.spectrogram_params)
#     if not args.no_val:
#         val_dataset = build_dataset(data_dir=os.path.join(args.data_dir, args.val_dir),
#                                     transforms=transforms,
#                                     data_mode=args.data_mode,
#                                     spectrogram_params=args.spectrogram_params)
#
#     # build data loaders
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     if not args.no_val:
#         val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)
#
#     # build model
#     model = build_model(name=args.model, num_classes=args.num_classes, pretrained=args.pretrained)
#
#     # setup device
#     device, device_ids = prepare_device(num_devices=args.num_devices)
#
#     # transfer model to device
#     model = model.to(device)
#
#     # print model summary
#     print(model)
#
#     # setup optimizer and loss function
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
#     lr_scheduler = LRScheduler(optimizer, name=args.scheduler, max_epochs=args.max_epochs, warmup_epochs=5)
#
#     # load checkpoint
#     if args.resume:
#         checkpoint = torch.load(args.resume, map_location=device)
#         args.start_epoch = checkpoint['epoch']
#         model.load_state_dict(checkpoint['model'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#         print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
#
#
# # todo:第五部分
# if __name__ == '__main__':
#     args = parse_args()
#
#     # set random seed for reproducibility
#     set_seed(42)
#
#     # prepare device
#     device, device_ids = prepare_device(args.n_gpu)
#
#     # create transforms
#     train_transforms = build_transforms(args, is_train=True)
#     val_transforms = build_transforms(args, is_train=False)
#
#     # create datasets and dataloaders
#     train_dataset = build_dataset(args, split='train', transforms=train_transforms)
#     val_dataset = build_dataset(args, split='val', transforms=val_transforms)
#
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
#                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
#                             num_workers=args.num_workers, pin_memory=True)
#
#     # build model architecture
#     model = build_model(args)
#
#     # define loss function
#     criterion = torch.nn.CrossEntropyLoss()
#
#     # define optimizer
#     optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
#                                 weight_decay=args.weight_decay)
#
#     # define lr scheduler
#     lr_scheduler = LRScheduler(optimizer, args.scheduler, args.learning_rate, len(train_loader), args.max_epochs)
#
#     # move model to device
#     model = model.to(device)
#
#     # multi-gpu training (should be after apex fp16 initialization)
#     if len(device_ids) > 1:
#         model = torch.nn.DataParallel(model, device_ids=device_ids)
#
#     # optionally resume from a checkpoint
#     if args.resume:
#         checkpoint = torch.load(args.resume, map_location='cpu')
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#         args.start_epoch = checkpoint['epoch'] + 1
#         print(f'Resuming training from epoch {args.start_epoch}')
#
#     # create logger
#     logger = Logger(args)
#
#     # train and evaluate the model
#     for epoch in range(args.start_epoch, args.max_epochs):
#         train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, lr_scheduler, epoch, device)
#         logger.add_scalar('train_loss', train_loss, epoch)
#         logger.add_scalar('train_acc', train_acc, epoch)
#
#         if not args.no_val:
#             val_loss, val_acc = validate(args, val_loader, model, criterion, epoch, device)
#             logger.add_scalar('val_loss', val_loss, epoch)
#             logger.add_scalar('val_acc', val_acc, epoch)
#
#         # save checkpoint
#         if epoch % args.checkpoint_interval == 0:
#             save_checkpoint(args, {
#                 'epoch': epoch,
#                 'model': args.model,
#                 'state_dict': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'lr_scheduler': lr_scheduler.state_dict(),
#                 'config': get_config_info(args),
#                 'train_acc': train_acc,
#                 'val_acc': val_acc,
#             })
#
#     # close logger
#     logger.close()
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from models import SimpleCNN
from trainer import Trainer
from visualization import plot_history, plot_confusion_matrix, plot_roc_curve, save_figure


def main(args):
    # 定义是否使用GPU
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    # 数据预处理，使用的transform均来自torchvision.transforms
    transform_train = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载数据集
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    # 定义数据加载器
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 创建模型实例并移动到GPU上
    model = SimpleCNN().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 创建训练器
    trainer = Trainer(model, criterion, optimizer, device)

    # 训练模型
    history = trainer.train(train_loader, test_loader, args.num_epochs)

    # 绘制训练曲线
    plot_history(history)

    # 在测试集上评估模型并绘制混淆矩阵和ROC曲线
    test_loss, test_acc, pred, target = trainer.test(test_loader)
    plot_confusion_matrix(pred, target, class_names=[str(i) for i in range(10)])
    fpr, tpr, roc_auc = trainer.get_roc_curve(pred, target, num_classes=10)
    plot_roc_curve(fpr, tpr, roc_auc)

    # 保存模型和可视化结果
    torch.save(model.state_dict(), 'model.pt')
    save_figure(plt.gcf(), 'train_history.png')
    save_figure(plt.gcf(), 'confusion_matrix.png')
    save_figure(plt.gcf(), 'roc_curve.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST classification')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--use_gpu', action='store_true', help='use GPU if available')
    args = parser.parse_args()

    main(args)
