import torch
from options import train_opts
from utils import logger, recorders, train_utils, test_utils
from datasets import custom_dataloader
from models import build_model

args = train_opts.TrainOpts().parse()
log  = logger.Logger(args)


def main(args):
    model = build_model(args, log)
    recorder = recorders.Records(records=None)

    train_loader, val_loader = custom_dataloader(args, log)

    for epoch in range(args.start_epoch, args.epochs+1):
        model.update_learning_rate()
        recorder.insert_record('train', 'lr', epoch, model.get_learning_rate())
        
        train_utils.train(args, log, train_loader, model, epoch, recorder)
        if epoch == 1 or (epoch % args.save_intv == 0): 
            model.save_checkpoint(epoch, recorder.records)
        log.plot_all_curves(recorder, 'train')

        if epoch % args.val_intv == 0:
            test_utils.test(args, log, 'val', val_loader, model, epoch, recorder)
            log.plot_all_curves(recorder, 'val')
        

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
