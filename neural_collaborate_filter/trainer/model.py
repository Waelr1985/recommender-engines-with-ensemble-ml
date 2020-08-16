
import os
import pandas as pd
import tensorflow as tf
import shutil
import datetime

# create dataset function
def create_dataset(path, column_names, label_name, defaults, batch_size, shuffle):
    """ Create tf.dataset from csv file.
    
    Args:
        path (str): Path to the csv file.
        column_names (list:str): List of string to specify which columns to use in dataset (including label).
        label_name (str): Column name for the label.
        defaults (list:str): List of string to set default values for columns.
        batch_size (str): Batchsize of the dataset.
        shuffle (bool): True for shuffling dataset and False otherwise.

    Returns:
        (tf.dataset): dataset used for training or testing
    """
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=path,
        select_columns=column_names,
        label_name=label_name,
        column_defaults = defaults,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=shuffle
    )
    return dataset

# model class
class NeuMF_Model(tf.keras.Model):
    """ The NeuMF_Model class. Takes user ids and item ids to predict ratings through matrix factorization and 
    multilayer perceptron. The user and item embeddings are user and item latent factors.

    Attributes:
        user_index_table (tf.lookup.StaticVocabularyTable): Table to transform user_id to user indices.
        item_index_table (tf.lookup.StaticVocabularyTable): Table to transform item_id to item indices.
        
        gmf_u_embed (tf.keras.layers.Embedding): User embedding layer for General Matrix Factorization.
        gmf_i_embed (tf.keras.layers.Embedding): Item embedding layer for General Matrix Factorization.
        mlp_u_embed (tf.keras.layers.Embedding): User embedding layer for Multilayer Perceptron.
        mlp_i_embed (tf.keras.layers.Embedding): Item embedding layer for Multilayer Perceptron.
        
        dense_1 (tf.keras.layers.Dense): First dense layer of multilayer perceptron stream.
        dense_2 (tf.keras.layers.Dense): Second dense layer of multilayer perceptron stream.
        dense_3 (tf.keras.layers.Dense): Third dense layer of multilayer perceptron stream.
        dense_4 (tf.keras.layers.Dense): Fourth dense layer of multilayer perceptron stream.
        output_layer (tf.keras.layers.Dense): Output dense layer for concat of mlp and gmf stream.
        
        mlp_concat (tf.keras.layers.Concatenate): Concatenate layer to combine user and item embedding in mlp stream.
        stream_concat (tf.keras.layers.Concatenate): Concatenate layer to combine mlp and gmf stream. 
    """
    def __init__(self, user_file_path, item_file_path, latent_num):
        """ init method for NeuMF_Model class
        
        Args:
            user_file_path (str): Path to txt file containing user ids.
            item_file_path (str): Path to txt file containing item ids.
            
        Returns:
            None
        """
        super(NeuMF_Model, self).__init__()
        self.user_index_table = self.create_lookup_table(user_file_path)
        self.item_index_table = self.create_lookup_table(item_file_path)
        
        user_id_size = self.get_size(user_file_path)
        item_id_size = self.get_size(item_file_path)
        self.gmf_u_embed = tf.keras.layers.Embedding(user_id_size, latent_num, name='gmf_u_embed')
        self.gmf_i_embed = tf.keras.layers.Embedding(item_id_size, latent_num, name='gmf_i_embed')
        self.mlp_u_embed = tf.keras.layers.Embedding(user_id_size, latent_num, name='mlp_u_embed')
        self.mlp_i_embed = tf.keras.layers.Embedding(item_id_size, latent_num, name='mlp_i_embed')

        self.dense_1 = tf.keras.layers.Dense(64, activation='relu', name='mlp_dense_1')
        self.dense_2 = tf.keras.layers.Dense(32, activation='relu', name='mlp_dense_2')
        self.dense_3 = tf.keras.layers.Dense(16, activation='relu', name='mlp_dense_3')
        self.dense_4 = tf.keras.layers.Dense(8, activation='relu', name='mlp_dense_4')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
        
        self.mlp_concat = tf.keras.layers.Concatenate(axis=1, name='mlp_concat')
        self.stream_concat = tf.keras.layers.Concatenate(axis=1, name='stream_concat')

    def create_lookup_table(self, file_path):
        """ create lookup table to translate ids to indices
        
        Args:
            file_path (str): Path to txt file containing ids.
            
        Returns:
            (tf.lookup.StaticVocabularyTable): The lookup table.
        """
        file_initializer = tf.lookup.TextFileInitializer(file_path, key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE, \
                            value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER, delimiter="\n")
        lookup_table = tf.lookup.StaticVocabularyTable(file_initializer, num_oov_buckets=1)
        return lookup_table
    
    def get_size(self, file_path):
        """ Get the total number of lines for a txt file, indicating the size of the column.
        
        Args:
            file_path (str): Path to txt file.
            
        Returns:
            (int): The total number of lines for the txt file.
        """
        id_text = tf.io.read_file(file_path)
        id_tensor = tf.strings.split(id_text, '\n')
        return id_tensor.shape[0]

    @tf.function
    def call(self, inputs, training):
        """The call method for NeuMF_Model class.

        Args:
            inputs (OrderedDict:tf.Tensor): OrderedDict of tensor containing user_id and item_id
        
        Returns:
            output (tf.Tensor): The predicted rating for the user and item combination.
        """
        user_id = inputs['user_id']
        item_id = inputs['item_id']

        # convert id to index
        user_index = self.user_index_table.lookup(user_id)
        item_index = self.item_index_table.lookup(item_id)

        # GMF stream
        gmf_u_latent = self.gmf_u_embed(user_index)
        gmf_i_latent = self.gmf_i_embed(item_index)

        # multiply latent factors
        gmf_out = gmf_u_latent * gmf_i_latent

        # MLP stream
        mlp_u_latent = self.mlp_u_embed(user_index)
        mlp_i_latent = self.mlp_i_embed(item_index)

        # concat latent factors and pass to dense layers
        mlp_concat_out = self.mlp_concat([mlp_u_latent, mlp_i_latent])
        dense_1_out = self.dense_1(mlp_concat_out)
        dense_2_out = self.dense_2(dense_1_out)
        dense_3_out = self.dense_3(dense_2_out)
        mlp_out = self.dense_4(dense_3_out)

        # concat GMF and MLP stream
        stream_concat_out = self.stream_concat([gmf_out, mlp_out])

        output = self.output_layer(stream_concat_out)
        return output
    
    
def save_latent_factors_to_bucket(col_name, col_path, tensor_weight, output_path, latent_num):
    """Store tensor weights as user or item latent factors to bucket in csv file.

    Args:
        col_name (str): Column name for the latent factors (user or item).
        col_path (str): Path to user or item ids.
        tensor_weight (tf.Tensor): Tensors of the embedding layer used as latent factors.
        output_path (str): Path to ouput file in bucket.
        latent_num (int): Number of latent factors

    Returns:
        None
    """
    id_tensors = tf.strings.split(tf.io.read_file(col_path), "\n")
    id_list = [tf.compat.as_str_any(x) for x in id_tensors.numpy()]

    latent_df = pd.DataFrame(tensor_weight)
    latent_df[col_name] = id_list

    key = range(latent_num * 2)
    value = ['{}_latent_'.format(col_name[0]) + str(x) for x in key]
    column_dict = dict(zip(key, value))
    
    latent_df = latent_df.rename(columns=column_dict)
    latent_df = latent_df[[col_name] + value]
    latent_df.to_csv("./latent.csv", index=False)
    
    script = "gsutil mv ./latent.csv {}".format(output_path)
    os.system(script)
    
    
def train_model_and_save_latent_factors(args):
    """ Train the NeuMF_Model and save embeddings as latent_factors to bucket in csv files.

    Args:
        args (dict): dict of arguments from task.py

    Returns:
        None
    """
    # create dataset
    column_name = ['user_id', 'item_id', 'rating']
    label_name = 'rating'
    defaults = ['unknown', 'unknown', 0.0]
    batch_size = args["batch_size"]
    train_path = args["train_data_path"]
    test_path = args["test_data_path"]
    
    train_dataset = create_dataset(train_path, column_name, label_name, defaults, batch_size, True)
    test_dataset = create_dataset(test_path, column_name, label_name, defaults, batch_size, False)
    
    # create model
    model = NeuMF_Model(args["user_id_path"], args["item_id_path"], args["latent_num"])
    
    # loss function and optimizers
    bc_loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    
    # loss metrics
    train_bc_loss = tf.keras.metrics.Mean(name='train_bc_loss')
    train_mae_loss = tf.keras.metrics.MeanAbsoluteError(name='train_mae_loss')
    train_rmse_loss = tf.keras.metrics.RootMeanSquaredError(name='train_rmse_loss')

    test_bc_loss = tf.keras.metrics.Mean(name='test_bc_loss')
    test_mae_loss = tf.keras.metrics.MeanAbsoluteError(name='test_mae_loss')
    test_rmse_loss = tf.keras.metrics.RootMeanSquaredError(name='test_rmse_loss')
    
    
    @tf.function
    def train_step(features, labels):
        """ Concrete function for train setp and update train metircs

        Args:
            features (OrderedDict:tf.Tensor): OrderedDict of tensor containing user_id and item_id as features.
            labels (tf.Tensor): labels (rating) of the training examples
            
        Returns:
            None
        """
        with tf.GradientTape() as tape:
            preds = model(features, training=True)
            bc_loss = bc_loss_object(labels, preds)
        gradients = tape.gradient(bc_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_bc_loss(bc_loss)
        train_mae_loss(labels, preds)
        train_rmse_loss(labels, preds)
    
    @tf.function
    def test_step(features, labels):
        """ Concrete function for test setp and update test metircs

        Args:
            features (OrderedDict:tf.Tensor): OrderedDict of tensor containing user_id and item_id as features.
            labels (tf.Tensor): labels (rating) of the training examples
            
        Returns:
            None
        """
        preds = model(features, training=False)
        bc_loss = bc_loss_object(labels, preds)
        test_bc_loss(bc_loss)
        test_mae_loss(labels, preds)
        test_rmse_loss(labels, preds)
    
    # custom train loop
    EPOCHS = args["epochs"]
    for epoch in range(EPOCHS):
        train_bc_loss.reset_states()
        train_mae_loss.reset_states()
        train_rmse_loss.reset_states()

        test_bc_loss.reset_states()
        test_mae_loss.reset_states()
        test_rmse_loss.reset_states()

        for features, labels in train_dataset:
            train_step(features, labels)

        for features, labels in test_dataset:
            test_step(features, labels)

        template = "Epoch {:d}, train [bc_loss: {:.5f}, mae_loss: {:.5f}, rmse_loss: {:.5f}], test [bc_loss: {:.5f}, mae_loss: {:.5f}, rmse_loss: {:.5f}]"
        print(template.format(epoch + 1, train_bc_loss.result(), train_mae_loss.result(), train_rmse_loss.result(), \
                                test_bc_loss.result(), test_mae_loss.result(), test_rmse_loss.result()))
        
    # export model
    EXPORT_PATH = os.path.join(args["output_dir"], datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    tf.saved_model.save(obj=model, export_dir=EXPORT_PATH)
    
    if args["save_latent_factors"]:
        # get embedding weights
        user_weight = tf.concat([model.gmf_u_embed.get_weights()[0], model.mlp_u_embed.get_weights()[0]], axis = 1).numpy()
        item_weight = tf.concat([model.gmf_i_embed.get_weights()[0], model.mlp_i_embed.get_weights()[0]], axis = 1).numpy()

        # store embedding weights to csv
        save_latent_factors_to_bucket('user_id', args["user_id_path"], user_weight, args["user_latent_path"], args["latent_num"])
        save_latent_factors_to_bucket('item_id', args["item_id_path"], item_weight, args["item_latent_path"], args["latent_num"])
