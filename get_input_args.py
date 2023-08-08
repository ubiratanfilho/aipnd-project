import argparse

def get_input_args_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type = str, default = './', 
                    help = 'path to the save checkpoints') 
    parser.add_argument('--arch', type = str, default = 'vgg13', 
                    help = 'CNN architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.001, 
                    help = 'Learning rate Hyperparameter')
    parser.add_argument('--hidden_units', type = int, default = 512, 
                    help = 'Hidden Units Hyperparameter')
    parser.add_argument('--epochs', type = int, default = 10, 
                    help = 'Epochs Hyperparameter')
    parser.add_argument('--gpu', type = bool, default = True, 
                    help = 'Use of GPU')
    
    return parser.parse_args()

def get_input_args_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type = str, default = 'flowers/test/1/image_06743.jpg')
    parser.add_argument('--top_k', type = int, default = 5)
    parser.add_argument('--gpu', type = bool, default = True, 
                    help = 'Use of GPU')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'categories dictionary')
  
    return parser.parse_args()