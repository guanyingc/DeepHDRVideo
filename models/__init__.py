import importlib
def find_model_from_string(model_name):
    modellib = importlib.import_module('models.%s' % model_name)
    model_class = getattr(modellib, model_name)
    return model_class

def build_model(args, log):
    print('Creating Model %s' % (args.model)) #TODO
    model_class = find_model_from_string(args.model)
    model = model_class(args, log)
    model.print_networks(log)
    return model

def get_option_setter(model_name):
    model_class = find_model_from_string(model_name)
    return model_class.modify_commandline_options
