import copy

datasets = {}

def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

def make(dataset_spec, args=None):
    dataset_args = copy.deepcopy(dataset_spec['args'])

    if args is not None:
        dataset_args.update(args)

    dataset = datasets[dataset_spec['name']](**dataset_args)
    
    return dataset
