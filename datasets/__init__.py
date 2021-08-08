import torch.utils.data
import importlib

def find_dataset_from_string(dataset_name):
    datasetlib = importlib.import_module('datasets.%s' % (dataset_name))
    dataset_class = getattr(datasetlib, dataset_name)
    return dataset_class

def get_option_setter(dataset_name):
    dataset_class = find_dataset_from_string(dataset_name)
    return dataset_class.modify_commandline_options

def custom_dataloader(args, log):
    log.print_write("=> fetching img pairs in %s" % (args.dataset))
    datasets = __import__('datasets.' + args.dataset)
    dataset_file = getattr(datasets, args.dataset)
    train_set = getattr(dataset_file, args.dataset)(args, 'train')
    val_set   = getattr(dataset_file, args.dataset)(args, 'val')

    if args.concat:
        log.print_write('****** Using cocnat data ******')
        log.print_write("=> fetching img pairs in '{}'".format(args.dataset2))
        if args.dataset2 == '':
            args.dataset2 = args.dataset
        dataset_class = find_dataset_from_string(args.dataset2)
        train_set2 = dataset_class(args, 'train')
        val_set2   = dataset_class(args, 'val')
        train_set  = torch.utils.data.ConcatDataset([train_set, train_set2])
        val_set    = torch.utils.data.ConcatDataset([val_set,   val_set2])

    log.print_write('Found Data:\t %d Train and %d Val' % (len(train_set), len(val_set)))
    log.print_write('\t Train Batch: %d, Val Batch: %d' % (args.batch, args.val_batch))

    use_gpu = len(args.gpu_ids) > 0
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch,
        num_workers=args.workers, pin_memory=use_gpu, shuffle=True)
    test_loader   = torch.utils.data.DataLoader(val_set , batch_size=args.val_batch,
        num_workers=args.workers, pin_memory=use_gpu, shuffle=False)
    return train_loader, test_loader

def benchmark_loader(args, log):
    log.print_write("=> fetching img pairs in 'data/%s'" % (args.benchmark))
    log.print_write("=> %s'" % (args.bm_dir))
    datasets = __import__('datasets.' + args.benchmark)
    dataset_file = getattr(datasets, args.benchmark)
    test_set = getattr(dataset_file, args.benchmark)(args, 'test')

    log.print_write('Found Benchmark Data: %d samples' % (len(test_set)))
    log.print_write('\t Test Batch %d' % (args.test_batch))

    use_gpu = len(args.gpu_ids) > 0
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch,
        num_workers=args.workers, pin_memory=use_gpu, shuffle=False)
    return test_loader
