import tensorflow as tf

from config import config
from src.ops import CNN
from src.ops import LSTM
from src.ops import highway
from src.reader import reader


class model(object):
    def __init__(self):
        self.reader = reader(config['data_dir'])

        # [batch_size, seq_len, word_len]
        self.char_inputs = tf.placeholder(
            tf.int32,
            shape=[None, config['max_seq_len'], config['max_word_len']],
            name='char_inputs'
        )

        # [batch_size, seq_len]
        self.word_inputs = tf.placeholder(
            tf.int32,
            shape=[None, config['max_seq_len']],
            name='word_inputs'
        )

        # [batch_size]
        self.label_inputs = tf.placeholder(
            tf.int32,
            shape=[None],
            name='label_inputs'
        )

        # [batch_size, seq_len, embedding_dim]
        embeds = self.get_embed(self.char_inputs, self.word_inputs)

        # [batch_size, feature_dim] * {1, 2, 3}
        before_mlp = []

        if config['use_direct_embed']:
            embed_dim = int(embeds.shape[1]) * int(embeds.shape[2])
            tmp = tf.reshape(embeds, (-1, embed_dim))
            # change the dimension of embedding
            # in case of the dim of embedding is much larger than
            # that of others
            with tf.name_scope('resize'):
                resize_W = tf.get_variable(name='W', shape=[tmp.shape[1], config['resize_to']])
                resize_b = tf.get_variable(name='b', shape=[config['resize_to']])
                before_mlp.append(tmp @ resize_W + resize_b)

        if config['use_LSTM']:
            before_mlp.append(self.LSTM(embeds))

        if config['use_CNN']:
            before_mlp.append(self.seq_CNN(embeds))

        # [batch_size, feature_dim]
        inputs = tf.concat(before_mlp, axis=1)

        with tf.variable_scope('mlp'):
            for idx, dim in enumerate(config['hidden_dims']):
                W = tf.get_variable(
                    name='W_%d' % idx,
                    shape=[inputs.shape[1], dim]
                )
                b = tf.get_variable(name='b_%d' % idx, shape=[dim])
                inputs = tf.sigmoid(inputs @ W + b)

        # [batch_size]
        self.pred = tf.reshape(inputs, shape=[-1])


    def LSTM(self, inputs):
        seq_LSTM = LSTM(
            int(inputs.shape[2]//2),
            config['fw_forget_bias'],
            config['bw_forget_bias'],
            'seq_LSTM'
        )
        ret = seq_LSTM(inputs, config['LSTM_layer_size'])
        dim = int(ret.shape[1]) * int(ret.shape[2])
        return tf.reshape(inputs, (-1, dim))

    def seq_CNN(self, inputs):

        seq_CNN = CNN(
            inputs.shape[2],
            config['seq_feature_maps'],
            config['seq_kernels'],
            1,
            'seq_CNN'
        )

        inputs = tf.expand_dims(inputs, -1)
        return seq_CNN(inputs)


    def get_embed(self, char_inputs, word_inputs):

        with tf.variable_scope('embed'):
            if config['use_char']:
                char_W = tf.get_variable(
                    name='char_embedding',
                    shape=[config['char_vocabulary_size'], config['char_embedding_dim']]
                )
                char_CNN = CNN(
                    config['char_embedding_dim'],
                    config['char_feature_maps'],
                    config['char_kernels'],
                    1,
                    'char_CNN'
                )
            if config['use_word']:
                word_W = tf.get_variable(
                    name='word_embedding',
                    shape=[config['max_word_num'], config['word_embedding_dim']]
                )

        char_indices = tf.unstack(char_inputs, config['max_seq_len'], axis=1)
        word_indices = tf.unstack(tf.expand_dims(word_inputs, -1), config['max_seq_len'], axis=1)

        ret = []

        for idx in range(config['max_seq_len']):
            combination = []
            char_index = char_indices[idx]
            word_index = word_indices[idx]

            if config['use_char']:
                char_embed = tf.nn.embedding_lookup(char_W, char_index)
                char_embed = char_CNN(tf.expand_dims(char_embed, -1))
                combination.append(char_embed)
            if config['use_word']:
                word_embed = tf.nn.embedding_lookup(word_W, word_index)
                word_embed = tf.reshape(word_embed, [-1, int(word_embed.shape[2])])
                combination.append(word_embed)
            assert len(combination) > 0
            embed = tf.concat(combination, axis=1)
            if config['use_highway']:
                if idx == 0:
                    highway_op = highway(
                        dim=int(embed.shape[1]),
                        layer_size=config['highway_layers'],
                        scope_name='highway'
                    )
                embed = highway_op(embed)

            ret.append(embed)

        return tf.stack(ret, axis=1)


    def train(self, log_path):

        with tf.variable_scope('evaluation'):
            accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.cast(self.pred * 2, tf.int32), self.label_inputs),
                tf.float32
            ))

            pred_pos = tf.cast(tf.reduce_sum(tf.cast(self.pred * 2, tf.int32)), tf.float32)
            pred_pos_ratio = pred_pos / config['batch_size']
            mean_pred = tf.reduce_mean(self.pred, name='mean_pred')
            all_pos = tf.reduce_sum(tf.cast(self.label_inputs, tf.float32))
            true_pos = tf.cast(tf.reduce_sum(tf.cast(
                2 * self.pred * tf.cast(self.label_inputs, tf.float32), tf.int32)), tf.float32)
            e_ = tf.constant(config['epsilon'])
            precision = true_pos / (pred_pos + e_)
            recall = true_pos / (all_pos + e_)
            f_measure = (2 * precision * recall) / (e_ + precision + recall)



#        loss = tf.reduce_mean(0.5 * ((self.label_inputs - self.pred) ** 2))
        with tf.variable_scope('train'):
            loss = tf.reduce_mean(
                tf.contrib.keras.backend.binary_crossentropy(
                    tf.cast(self.label_inputs, tf.float32), self.pred
                ), name='loss_function'
            )
            optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), _) for grad,_ in gvs if grad is not None]
            train_op = optimizer.apply_gradients(capped_gvs)

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('pred_pos_ratio', pred_pos_ratio)
        tf.summary.scalar('precision', precision)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('f_measure', f_measure)
        tf.summary.scalar('mean_pred', mean_pred)
        #tf.summary.histogram('pred', self.pred)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(log_path, sess.graph)
            sess.run(tf.global_variables_initializer())
            for n_ in range(config['n_epoch']):
                print('%d epoch...' % n_)
                gen = self.reader.trainDataGenerator(config['batch_size'])
                for [charIn, wordIn, labelIn], ratio, curr in gen:
                    if curr % config['print_iteration'] == 0:
                        batch_loss, batch_acc = sess.run([loss, accuracy], feed_dict={
                            self.char_inputs: charIn,
                            self.word_inputs: wordIn,
                            self.label_inputs: labelIn
                        })
                        print('process: %f' % ratio)
                        print('batch loss: %f\nbatch accuracy: %f\n' % (batch_loss, batch_acc))
                    sess.run(train_op, feed_dict={
                        self.char_inputs: charIn,
                        self.word_inputs: wordIn,
                        self.label_inputs: labelIn
                    })
                print('validating ...')


                (charIn, wordIn, labelIn) = self.reader.valDataGenerator()
                merged_ = sess.run(
                    merged,
                    feed_dict={
                        self.char_inputs: charIn,
                        self.word_inputs: wordIn,
                        self.label_inputs: labelIn
                    }
                )
                train_writer.add_summary(merged_, n_)


if __name__ == '__main__':
    #model().train()
    pass

