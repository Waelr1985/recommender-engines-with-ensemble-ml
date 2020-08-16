
import argparse
import tensorflow as tf
from trainer import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        help="job dir to store training outputs and other data",
        required=True
    )
    
    parser.add_argument(
        "--train_data_path",
        help="path to import train data",
        required=True
    )
    
    parser.add_argument(
        "--test_data_path",
        help="path to import test data",
        required=True
    )
    
    parser.add_argument(
        "--output_dir",
        help="output dir to export checkpoints or trained model",
        required=True
    )
    
    parser.add_argument(
        "--batch_size",
        help="batch size for training",
        type=int,
        default=2048
    )
    
    parser.add_argument(
        "--epochs",
        help="number of epochs for training",
        type=int,
        default=1
    )
    
    parser.add_argument(
        "--latent_num",
        help="number of latent factors for gmf and mlp each",
        type=int,
        default=8
    )
    
    parser.add_argument(
        "--user_id_path",
        help="path to import user_id_list.txt",
        required=True
    )
    
    parser.add_argument(
        "--item_id_path",
        help="path to import item_id_list.txt",
        required=True
    )
    
    parser.add_argument(
        "--user_latent_path",
        help="output path to save user latent factors",
        default="./"
    )
    
    parser.add_argument(
        "--item_latent_path",
        help="output path to save item latent factors",
        default="./"
    )
    
    parser.add_argument(
        "--save_latent_factors",
        help="set to save latent factors",
        default=False,
        action="store_true"
    )
    
    args = parser.parse_args()
    args = args.__dict__
    
    model.train_model_and_save_latent_factors(args)
