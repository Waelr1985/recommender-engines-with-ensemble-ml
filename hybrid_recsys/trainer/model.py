
import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import shutil
import datetime

# create dataset function
def create_dataset(path, column_name, label_name, defaults, batch_size, shuffle):
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
        file_pattern = path,
        select_columns=column_name,
        label_name=label_name,
        column_defaults = defaults,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=shuffle
    )
    return dataset

# model class
class Hybrid_Recsys_Model(tf.keras.Model):
    """ The Hybrid_Recsys_Model class. For training, the model takes input features and predict the probability for index of next item id. 
    For serving, the model takes input features and predict the next item id directly.

    Attributes:
        user_id_hash (tf.feature_column.categorical_column_with_hash_bucket): hash bucket column for user_id.
        item_id_hash (tf.feature_column.categorical_column_with_hash_bucket): hash bucket column for item_id.
        author_hash (tf.feature_column.categorical_column_with_hash_bucket): hash bucket column for author.
        device_brand_hash (tf.feature_column.categorical_column_with_hash_bucket): hash bucket column for device_brand.
        
        user_id_embed (tf.feature_column.embedding_column): embedding column for user_id.
        item_id_embed (tf.feature_column.embedding_column): embedding column for item_id.
        author_embed (tf.feature_column.embedding_column): embedding column for author.
        device_brand_embed (tf.feature_column.embedding_column): embedding column for device_brand.
        
        author_vocab (tf.feature_column.categorical_column_with_vocabulary_list): vocabulary list column for author.
        article_year_vocab (tf.feature_column.categorical_column_with_vocabulary_list): vocabulary list column for article_year.
        article_month_vocab (tf.feature_column.categorical_column_with_vocabulary_list): vocabulary list column for article_month.
        category_vocab (tf.feature_column.categorical_column_with_vocabulary_list): vocabulary list column for category.
        device_brand_vocab (tf.feature_column.categorical_column_with_vocabulary_list): vocabulary list column for device_brand.
        
        author_indicator (tf.feature_column.indicator_column): indicator colunm for author.
        cross_date_indicator (tf.feature_column.indicator_column): indicator colunm for crossing article_year and article_month.
        category_indicator (tf.feature_column.indicator_column): indicator colunm for category.
        device_brand_indicator (tf.feature_column.indicator_column): indicator colunm for device_brand.
        
        cross_date (tf.feature_column.crossed_column): crossed column for crossing rticle_year and article_month.
        
        u_latent_numeric (list:tf.feature_column.numeric_column): list of numeric columns for user latent factors.
        i_latent_numeric (list:tf.feature_column.numeric_column): list of numeric columns for item latent factors
        
        feature_columns_d (list:tf.feature_column): list of feature columns for deep features.
        feature_columns_w (list:tf.feature_column): list of feature columns for wide features.
        
        feature_layer_d (tf.keras.layers.DenseFeatures): dense layer for deep feature column.
        feature_layer_d (tf.keras.layers.DenseFeatures): dense layer for wide feature column.
        
        text_embed_layer (hub.KerasLayer): tenseorflow hub layer (NNLM) for text embedding
        
        dense_1 (tf.keras.layers.Dense): first dense layer for deep network
        dense_2 (tf.keras.layers.Dense): second dense layer for deep network
        dense_3 (tf.keras.layers.Dense): third dense layer for deep network
        dense_4 (tf.keras.layers.Dense): output layer which takes wide and deep networks and predict index for next item.
        
        item_id_table (tf.lookup.StaticHashTable): table for converting index to ids
    """
    def __init__(self, item_id_path, author_path, category_path, device_brand_path, article_year_path, article_month_path, latent_num):
        """ init method for Hybrid_Recsys_Model class
        
        Args:
            item_id_path (str): Path to txt file containing unique item ids.
            author_path (str): Path to txt file containing unique authors.
            category_path (str): Path to txt file containing unique categories.
            device_brand_path (str): Path to txt file containing unique device_brands.
            article_year_path (str): Path to txt file containing unique article_years.
            article_month_path (str): Path to txt file containing unique article_months.
            latent_num (int): Number of laten factors (gmf and mlp stream each) for representing user ids and item ids.
            
        Returns:
            None
        """
        super(Hybrid_Recsys_Model, self).__init__()
        # user_id embed
        self.user_id_hash = tf.feature_column.categorical_column_with_hash_bucket('user_id', 100)
        self.user_id_embed = tf.feature_column.embedding_column(categorical_column=self.user_id_hash, dimension=5)

        # item_id embed
        self.item_id_hash = tf.feature_column.categorical_column_with_hash_bucket('item_id', 200)
        self.item_id_embed = tf.feature_column.embedding_column(categorical_column=self.item_id_hash, dimension=20)

        # author embed
        self.author_hash = tf.feature_column.categorical_column_with_hash_bucket('author', 20)
        self.author_embed = tf.feature_column.embedding_column(categorical_column=self.author_hash, dimension=10)
        
        # author indicator
        self.author_vocab = tf.feature_column.categorical_column_with_vocabulary_list('author', self.get_list(author_path))
        self.author_indicator = tf.feature_column.indicator_column(self.author_vocab)

        # cross article year, month
        self.article_year_vocab = tf.feature_column.categorical_column_with_vocabulary_list('article_year', self.get_list(article_year_path))
        self.article_month_vocab = tf.feature_column.categorical_column_with_vocabulary_list('article_month', self.get_list(article_month_path))
        
        self.cross_date = tf.feature_column.crossed_column([self.article_year_vocab, self.article_month_vocab], \
                                                           self.get_size(article_year_path) * self.get_size(article_month_path))
        self.cross_date_indicator = tf.feature_column.indicator_column(self.cross_date)

        # category indicator
        self.category_vocab = tf.feature_column.categorical_column_with_vocabulary_list('category', self.get_list(category_path))
        self.category_indicator = tf.feature_column.indicator_column(self.category_vocab)

        # device_brand embed
        self.device_brand_hash= tf.feature_column.categorical_column_with_hash_bucket('device_brand', 10)
        self.device_brand_embed = tf.feature_column.embedding_column(categorical_column=self.device_brand_hash, dimension=5)

        # device_brand indicator
        self.device_brand_vocab = tf.feature_column.categorical_column_with_vocabulary_list('device_brand', self.get_list(device_brand_path))
        self.device_brand_indicator = tf.feature_column.indicator_column(self.device_brand_vocab)

        # user item latent factors
        self.u_latent_numeric = [tf.feature_column.numeric_column(key="user_latent_" + str(i)) for i in range(2 * latent_num)]
        self.i_latent_numeric =  [tf.feature_column.numeric_column(key="item_latent_" + str(i)) for i in range(2 * latent_num)]

        # combine feature columns to feature layer
        self.feature_columns_d =  [self.user_id_embed, self.item_id_embed, self.author_embed, self.device_brand_embed] \
                                    + self.u_latent_numeric + self.i_latent_numeric
        self.feature_layer_d = tf.keras.layers.DenseFeatures(self.feature_columns_d)
        
        self.feature_columns_w = [self.author_indicator, self.cross_date_indicator, self.category_indicator, self.device_brand_indicator]
        self.feature_layer_w = tf.keras.layers.DenseFeatures(self.feature_columns_w)

        # title tf_hub nnlm embedding
        self.text_embed_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-de-dim50-with-normalization/2", dtype=tf.string)
       
        # dense
        self.dense_1 = tf.keras.layers.Dense(200, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(100, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(50, activation='relu')
        self.dense_4 = tf.keras.layers.Dense(self.get_size(item_id_path) + 1, activation='softmax')

        # item_id lookup table
        self.item_id_table = self.create_item_id_table(item_id_path)
    
    @tf.function
    def call(self, inputs):
        """The call method for NeuMF class.

        Args:
            inputs (OrderedDict:tf.Tensor): OrderedDict of input feature tensor.
        
        Returns:
            output (tf.Tensor): The predicted probability for index of next item id.
        """
        # wide, deep feature columns
        feature_cols_d = self.feature_layer_d(inputs)
        feature_cols_w = self.feature_layer_w(inputs)
        
        # title embedding
        title = inputs['title']
        title_embed = self.text_embed_layer(title)
        
        # deep network
        concat_1 = tf.concat([feature_cols_d, title_embed], axis=1)
        dense_1_out = self.dense_1(concat_1)
        dense_2_out = self.dense_2(dense_1_out)
        dense_3_out = self.dense_3(dense_2_out)
        
        # combine wide, and deep layers
        concat_2 = tf.concat([dense_3_out, feature_cols_w], axis=1)

        output = self.dense_4(concat_2)
        return output
    
    
    latent_num = 10
    key = ["user_id", "item_id", "title", "author", "category", "device_brand", "article_year", "article_month"] + \
            ["user_latent_{}".format(i) for i in range(2*latent_num)] + ["item_latent_{}".format(i) for i in range(2*latent_num)]
    value = [tf.TensorSpec([None], dtype=tf.string, name="user_id"),
                tf.TensorSpec([None], dtype=tf.string, name="item_id"), \
                tf.TensorSpec([None], dtype=tf.string, name="title"), \
                tf.TensorSpec([None], dtype=tf.string, name="author"), \
                tf.TensorSpec([None], dtype=tf.string, name="category"), \
                tf.TensorSpec([None], dtype=tf.string, name="device_brand"), \
                tf.TensorSpec([None], dtype=tf.string, name="article_year"), \
                tf.TensorSpec([None], dtype=tf.string, name="article_month")] + \
            [tf.TensorSpec([None], dtype=tf.float32, name="user_latent_{}".format(i)) for i in range(2*latent_num)] + \
            [tf.TensorSpec([None], dtype=tf.float32, name="item_latent_{}".format(i)) for i in range(2*latent_num)]
    signature_dict = dict(zip(key, value))
    @tf.function(input_signature=[signature_dict])
    def my_serve(self, x):
        """The serving method for Hybrid class.

        Args:
            inputs (OrderedDict:tf.Tensor): OrderedDict of input feature tensor.
        
        Returns:
            output (tf.Tensor): The predicted next item id.
        """
        pred = self.__call__(x)
        values, indices = tf.math.top_k(pred, k=10)
        item_ids = self.item_id_table.lookup(indices)
        return {"top_k_item_id": item_ids}
    
    def get_size(self, file_path):
        """Returns total number of lines in the txt file.

        Args:
            file_path (str): Path to txt file.
        
        Returns:
            size (int): total number of lines in the txt file.
        """
        id_tensors = tf.strings.split(tf.io.read_file(file_path), '\n')
        return id_tensors.shape[0]
    
    def get_list(self, file_path):
        """Extract content from txt file and store into list. Each line in txt file corresponds to an elemnt in list.

        Args:
            file_path (str): Path to txt file.
        
        Returns:
            id_list (int): list of values from txt file.
        """
        id_tensors = tf.strings.split(tf.io.read_file(file_path), '\n')
        id_list = [tf.compat.as_str_any(x) for x in id_tensors.numpy()]
        return id_list
    
    def create_item_id_table(self, file_path):
        """ create lookup table to translate item index to item id.
        
        Args:
            file_path (str): Path to txt file containing item ids.
            
        Returns:
            (tf.lookup.StaticVocabularyTable): The lookup table.
        """
        values = tf.strings.split(tf.io.read_file(file_path), '\n')
        keys = tf.range(values.shape[0], dtype=tf.int32)
        initializer = tf.lookup.KeyValueTensorInitializer(keys=keys, values=values, key_dtype=tf.int32, value_dtype=tf.string)
        table = tf.lookup.StaticHashTable(initializer, default_value='unknown')
        return table
        
        
    
def train_and_export_model(args):
    """ Train the Hybrid_Recsys_Model and export model to bucket.

    Args:
        args (dict): dict of arguments from task.py

    Returns:
        None
    """
    # create dataset
    feature_col = ['user_id', 'item_id', 'title', 'author', 'category', 'device_brand', 'article_year', 'article_month', 'next_item_id']
    u_latent_col = ["user_latent_{}".format(i) for i in range(2 * args["latent_num"])]
    i_latent_col = ["item_latent_{}".format(i) for i in range(2 * args["latent_num"])]
    column_name = feature_col + u_latent_col + i_latent_col
    
    label_name = 'next_item_id'
    defaults = ['unknown'] * len(feature_col) + [0.0] * (len(u_latent_col) + len(i_latent_col))
    batch_size = args["batch_size"]
    train_path = args["train_data_path"]
    test_path = args["test_data_path"]
    
    train_dataset = create_dataset(train_path, column_name, label_name, defaults, batch_size, True)
    test_dataset = create_dataset(test_path, column_name, label_name, defaults, batch_size, False)
    
    # create model
    model = Hybrid_Recsys_Model(args["item_id_path"], args["author_path"], args["category_path"], args["device_brand_path"], \
                                args["article_year_path"], args["article_month_path"], args["latent_num"])
    
    # loopup table (convvert id to index)
    item_index_initializer = tf.lookup.TextFileInitializer(args["item_id_path"], key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE, \
                            value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER, delimiter="\n")
    item_index_table = tf.lookup.StaticVocabularyTable(item_index_initializer, num_oov_buckets=1)
    
    
    # loss function and optimizers
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    
    # loss metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_top_10_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='train_top_10_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_top_10_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='test_top_10_accuracy')
    
    # tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './logs/gradient_tape/' + current_time + '/train'
    test_log_dir = './logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    
    @tf.function
    def train_step(features, labels):
        """ Concrete function for train setp and update train metircs

        Args:
            features (OrderedDict:tf.Tensor): OrderedDict of tensor containing input  features.
            labels (tf.Tensor): labels indicating the next_item id

        Returns:
            None
        """
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            label_indicies = item_index_table.lookup(labels)
            loss = loss_object(label_indicies, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(label_indicies, predictions)
        train_top_10_accuracy(label_indicies, predictions)
    
    @tf.function
    def test_step(features, labels):
        """ Concrete function for test setp and update test metircs

        Args:
            features (OrderedDict:tf.Tensor): OrderedDict of tensor containing input features.
            labels (tf.Tensor): labels indicating the next_item id

        Returns:
            None
        """
        predictions = model(features, training=False)
        label_indicies = item_index_table.lookup(labels)
        loss = loss_object(label_indicies, predictions)

        test_loss(loss)
        test_accuracy(label_indicies, predictions)
        test_top_10_accuracy(label_indicies, predictions)
    
    # custom train loop
    EPOCHS = args["epochs"]

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        train_top_10_accuracy.reset_states()

        test_loss.reset_states()
        test_accuracy.reset_states()
        test_top_10_accuracy.reset_states()

        for features, labels in train_dataset:
            train_step(features, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            tf.summary.scalar('top_10_accuracy', train_top_10_accuracy.result(), step=epoch)

        for features, labels in test_dataset:
            test_step(features, labels)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
            tf.summary.scalar('top_10_accuracy', test_top_10_accuracy.result(), step=epoch)

        template = 'Epoch {:d}, train[loss: {:.6f}, acc: {:.6f}, top_10_acc: {:.6f}], Test[loss: {:.6f}, acc: {:.6f}, top_10_acc: {:.6f}]'

        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              train_top_10_accuracy.result() * 100,

                              test_loss.result(),
                              test_accuracy.result() * 100,
                              test_top_10_accuracy.result() * 100,
                              ))
    
    # exprot tensorboard log
    if args["save_tb_log_to_bucket"]:
        script = "gsutil cp -r ./logs {}".format(args["bucket_tb_log_path"])
        os.system(script)
    
    # export model
    EXPORT_PATH = os.path.join(args["output_dir"], datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    tf.saved_model.save(obj=model, export_dir=EXPORT_PATH, signatures={'serving_default': model.my_serve})
    
