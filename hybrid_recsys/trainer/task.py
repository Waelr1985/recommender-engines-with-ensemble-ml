
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
        help="number of latent factors for gmf and mlp",
        type=int,
        default=10
    )

    parser.add_argument(
        "--item_id_path",
        help="path to import item_id_list.txt",
        required=True
    )
    
    parser.add_argument(
        "--author_path",
        help="path to import author_list.txt",
        required=True
    )
    
    parser.add_argument(
        "--category_path",
        help="path to import category_list.txt",
        required=True
    )
    
    parser.add_argument(
        "--device_brand_path",
        help="path to import device_brand_list.txt",
        required=True
    )
    parser.add_argument(
        "--article_year_path",
        help="path to import article_year_list.txt",
        required=True
    )
    
    parser.add_argument(
        "--article_month_path",
        help="path to import article_month_list.txt",
        required=True
    )
    
    parser.add_argument(
        "--save_tb_log_to_bucket",
        help="set to save tensorboard logs in bucket",
        default=False,
        action="store_true"
    )
    
    parser.add_argument(
        "--bucket_tb_log_path",
        help="path to store tensorboard in gcp bucket",
        default="gs://hybrid-recsys-gcp-bucket/tensorboard_log/ "
    )
    
    
    args = parser.parse_args()
    args = args.__dict__

    model.train_and_export_model(args)
