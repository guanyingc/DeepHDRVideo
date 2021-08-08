import torch
from models import model_utils
from utils import time_utils 


def get_itervals(args, split):
    if split not in ['train', 'val', 'test']:
        split = 'test'
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    stop_iters = args_var['max_'+split+'_iter']
    return disp_intv, save_intv, stop_iters


def test(args, log, split, loader, model, epoch, recorder):
    model.eval()
    log.print_write('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv, stop_iters = get_itervals(args, split)
    res = []
    with torch.no_grad():
        for i, sample in enumerate(loader):
            timer.update_time('ToCPU')

            data = model.parse_data(sample) 
            timer.update_time('ToGPU')

            model.epoch = epoch
            model.iter = i
            pred = model.forward(split=split); 
            timer.update_time('Forward')

            loss = model.get_loss_terms()

            if loss != None: 
                recorder.update_iter(split, loss.keys(), loss.values())
        
            if args.save_records:
                records, iter_res = model.prepare_records()
                recorder.update_iter(split, records.keys(), records.values())

                res.append(iter_res)

            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                log.print_iters_summary(opt)

            if iters % save_intv == 0:
                if not args.disable_save:
                    visuals = model.prepare_visual() 

                    nrow = visuals[0].shape[0] # batch
                    log.save_img_results(visuals, split, epoch, iters, nrow=nrow)

                log.plot_curves(recorder, split, epoch=epoch, intv=disp_intv)
                                
                if (hasattr(args, 'save_detail') and args.save_detail) and split == 'test':
                    model.save_visual_details(log, split, epoch, i)

                timer.update_time('Visualization')

            if stop_iters > 0 and iters >= stop_iters: break

    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    summary_str = log.print_epoch_summary(opt)
        
    if args.save_records:
        log.save_txt_result(res, split, epoch, summary_str)
