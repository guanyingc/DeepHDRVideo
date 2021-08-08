import torch
from options import run_model_opts
from utils import logger, recorders, test_utils
from datasets import benchmark_loader
from models import build_model

args = run_model_opts.RunModelOpts().parse()
log  = logger.Logger(args)

def main(args):
    test_loader = benchmark_loader(args, log)
    model = build_model(args, log)
    recorder = recorders.Records()

    test_utils.test(args, log, 'test', test_loader, model, 1, recorder)
    log.plot_curves(recorder, 'test')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
