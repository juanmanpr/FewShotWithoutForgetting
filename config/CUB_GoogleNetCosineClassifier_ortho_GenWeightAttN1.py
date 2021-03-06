config = {}
# set the parameters related to the training and testing set

nKbase = 100
nKnovel = 5
nExemplars = 1

data_train_opt = {}
data_train_opt['nKnovel'] = nKnovel
data_train_opt['nKbase'] = -1
data_train_opt['nExemplars'] = nExemplars
data_train_opt['nTestNovel'] = nKnovel * 3
data_train_opt['nTestBase'] = nKnovel * 3 
data_train_opt['batch_size'] = 2
data_train_opt['epoch_size'] = data_train_opt['batch_size'] * 1000

data_test_opt = {}
data_test_opt['nKnovel'] = nKnovel
data_test_opt['nKbase'] = nKbase
data_test_opt['nExemplars'] = nExemplars
data_test_opt['nTestNovel'] = 12 * data_test_opt['nKnovel']
data_test_opt['nTestBase'] = 12 * data_test_opt['nKnovel']
data_test_opt['batch_size'] = 1
data_test_opt['epoch_size'] = 2000

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt

config['max_num_epochs'] = 60

networks = {}
net_optionsF = {'userelu': False, 'in_planes':3, 'out_planes':[64,64,128,128], 'num_stages':4}
pretrainedF = './experiments/CUB_GoogleNetCosineClassifier_ortho/feat_model_net_epoch13_'
networks['feat_model'] = {'def_file': 'architectures/googlenet_master.py', 'pretrained': pretrainedF, 'opt': net_optionsF, 'optim_params': None}

net_optim_paramsC = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 1e-4, 'nesterov': True, 'LUT_lr':[(20, 0.1),(40, 0.006),(50, 0.0012),(60, 0.00024)], 'reg': 'ortho', 'ortho_lambda': 1e-4}
pretrainedC = './experiments/CUB_GoogleNetCosineClassifier_ortho/classifier_net_epoch13_'
net_optionsC = {'classifier_type': 'cosine', 'weight_generator_type': 'attention_based', 'nKall': nKbase, 'nFeat':1024, 'scale_cls': 10, 'scale_att': 10.0}
networks['classifier'] = {'def_file': 'architectures/ClassifierWithFewShotGenerationModule.py', 'pretrained': pretrainedC, 'opt': net_optionsC, 'optim_params': net_optim_paramsC}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions

config['algorithm_type'] = 'FewShot'
