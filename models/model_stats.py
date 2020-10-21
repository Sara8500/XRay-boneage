
def get_num_params_module(module):
    """return number of parameters stored on gpu for given module"""

    num_params = 0
    for mod in module.children():
        num_params += get_num_params_module(mod)

    if hasattr(module, 'weight') and module.weight is not None:
        num_params += module.weight.numel()
    if hasattr(module, 'bias') and module.bias is not None:
        num_params += module.bias.numel()

    return num_params

def get_num_params(model):
    """return number of parameters stored on gpu for given model"""

    num_params = 0
    for module in model.module.children():
        num_params += get_num_params_module(module)

    return num_params
