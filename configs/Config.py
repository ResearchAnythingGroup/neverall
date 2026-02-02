import os


def replace_path_sep_to_system_sep(path):
    seps = ['/', '\\']
    system_sep = os.sep

    path = path.replace(seps[0], system_sep)
    path = path.replace(seps[1], system_sep)
    return path


class Constant:
    # dataset
    CIFAR10 = 'cifar-10'
    CIFAR100 = 'cifar-100'
    FLOWER102 = 'flower-102'
    PET37 = 'pet-37'
    FOOD101 = 'food-101'
    COUNTRY211 = 'country-211'
    IMAGENETTE = 'imagenette'
    GTSRB = 'gtsrb'
    STL10 = 'stl10'

    # dataloader
    # original data
    TRAIN_DATA = 'train'
    TEST_DATA = 'test'
    TEST_DATA_ONLY_FORGET = 'test_forget'
    TEST_DATA_WITHOUT_FORGET = 'test_retain'
    # generate data
    FORGET_DATA = 'forget'  # from train data only contain forget classes (exp:10%)
    RETAIN_DATA = 'retain'  # from train data without forget classes
    #  INC_DATA_FT = INC_DATA_TEST + INC_DATA_FORGET + INC_DATA_TRANSFORM
    INC_DATA_FT = 'inc_ft'  # all the incremental data for finetune
    INC_DATA_TEST = 'inc_test'  # from TEST_DATA (25%)
    INC_DATA_FORGET = 'inc_forget'  # from FORGET_DATA (50%)
    INC_DATA_TRANSFORM = 'inc_transform'  # INC_DATA_FORGET to transform

    # device
    CUDA = 'cuda'
    CPU = 'cpu'

    # datapath suffix
    WWW = 'www'
    TRAIN = 'train'
    PRETRAIN = 'pretrain'
    INC_SUFFIX = 'ul_ft'
    NRC_SUFFIX = "ul_ft_neverecall"

    BAR = 'bar'
    CMT = 'cmt'  # confusion matrix
    TSNE = 'tsne'  # t-SNE

    ULM = 'ulm'  # unlearn model
    NRM = 'nrm'  # neverecall model
    INCM = 'incm'  # incremental model
    ALL = 'all'

    # ablation type
    SSR = 'SSR'
    SSGS = 'SSGS'
    SSPR = 'SSPR'
