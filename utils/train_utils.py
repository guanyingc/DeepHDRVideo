import torch
from models import model_utils
from utils import time_utils


def train(args, log, loader, model, epoch, recorder):
    model.train()
    log.print_write('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);
    for i, sample in enumerate(loader):
        timer.update_time('ToCPU')
        model.parse_data(sample) 
        timer.update_time('ToGPU')

        model.epoch = epoch
        model.iter = i
        pred = model.forward(split='train'); 
        timer.update_time('Forward')

        model.optimize_weights()
        timer.update_time('Backward')
        
        loss = model.get_loss_terms()
        recorder.update_iter('train', loss.keys(), loss.values())

        iters = i + 1
        if iters % args.train_disp == 0:
            opt = {'split':'train', 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                    'timer':timer, 'recorder': recorder}
            log.print_iters_summary(opt)

        if iters % args.train_save == 0:
            records, _ = model.prepare_records()
            visuals = model.prepare_visual() 
            recorder.update_iter('train', records.keys(), records.values())
            nrow = min(visuals[0].shape[0], 32)

            if epoch == 1 or iters % (2 * args.train_save) == 0:
                if not args.disable_save:
                    log.save_img_results(visuals, 'train', epoch, iters, nrow=nrow)

            log.plot_curves(recorder, 'train', epoch=epoch, intv=args.train_disp)

            timer.update_time('Visualization')

        if args.max_train_iter > 0 and iters >= args.max_train_iter: break
    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.print_epoch_summary(opt)

