import tensorflow as tf
from utils import *
from Dice import dice

class Model(object):
    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
    # input data
        with tf.name_scope('Inputs'):
            # label
            self.y_ph = tf.placeholder(tf.float32, [None, None], name='y_ph')
            
            # learning rate
            self.lr = tf.placeholder(tf.float64, [], name='lr')
            
            # ad static feature
            self.adgroup_batch_ph = tf.placeholder(tf.int32, [None, ], name='adgroup_batch_ph')
            self.member_batch_ph = tf.placeholder(tf.int32, [None, ], name='member_batch_ph')
            self.campaign_batch_ph = tf.placeholder(tf.int32, [None, ], name='campaign_batch_ph')
            self.item_batch_ph = tf.placeholder(tf.int32, [None, ], name='item_batch_ph')
            self.item_price_batch_ph = tf.placeholder(tf.float32, [None, ], name='item_price_batch_ph')
            self.cate_batch_ph = tf.placeholder(tf.int32, [None, ], name='cate_batch_ph')
            self.commodity_batch_ph = tf.placeholder(tf.int32, [None, ], name='commodity_batch_ph')
            self.node_batch_ph = tf.placeholder(tf.int32, [None, ], name='node_batch_ph')

            # effect data
            self.effect_batch_his_id_ph = tf.placeholder(tf.int32, [None, None, None], name='effect_batch_his_id_ph')
            self.effect_batch_his_val_ph = tf.placeholder(tf.float32, [None, None, None], name='effect_batch_his_val_ph')
            self.effect_batch_his_mask_ph = tf.placeholder(tf.float32, [None, None], name='effect_batch_his_mask_ph')

            # pos data
            self.pos_batch_his_id_ph = tf.placeholder(tf.int32, [None, None, None], name='pos_batch_his_id_ph')
            self.pos_batch_his_val_ph = tf.placeholder(tf.float32, [None, None, None], name='pos_batch_his_val_ph')
            self.pos_batch_his_mask_ph = tf.placeholder(tf.float32, [None, None], name='pos_batch_his_mask_ph')

            # direct data
            self.direct_batch_his_id_ph = tf.placeholder(tf.int32, [None, None, None], name='direct_batch_his_id_ph')
            self.direct_batch_his_val_ph = tf.placeholder(tf.float32, [None, None, None], name='direct_batch_his_val_ph')
            self.direct_batch_his_mask_ph = tf.placeholder(tf.float32, [None, None, None], name='direct_batch_his_mask_ph')

            # direct action
            self.direct_action_batch_his_id_ph = tf.placeholder(tf.int32, [None, None, None], name='direct_action_batch_his_id_ph')
            self.direct_action_batch_his_val_ph = tf.placeholder(tf.float32, [None, None, None], name='direct_action_batch_his_val_ph')
            self.direct_action_batch_his_mask_ph = tf.placeholder(tf.float32, [None, None, None], name='direct_action_batch_his_mask_ph')

            # pos action
            self.pos_action_batch_his_id_ph = tf.placeholder(tf.int32, [None, None, None], name='pos_action_batch_his_id_ph')
            self.pos_action_batch_his_val_ph = tf.placeholder(tf.float32, [None, None, None], name='pos_action_batch_his_val_ph')
            self.pos_action_batch_his_mask_ph = tf.placeholder(tf.float32, [None, None, None], name='pos_action_batch_his_mask_ph')

    # embedding layer
        with tf.name_scope('Embedding_layer'):
            self.adgroup_embedding_var = tf.get_variable("adgroup_embedding_var", [n_adgroup, EMBEDDING_DIM // 3])
            tf.summary.histogram("adgroup_embedding_var", self.adgroup_embedding_var)
            self.adgroup_batch_embedded = tf.nn.embedding_lookup(self.adgroup_embedding_var, self.adgroup_batch_ph)

            self.member_embedding_var = tf.get_variable("member_embedding_var", [n_member, EMBEDDING_DIM // 3])
            tf.summary.histogram("member_embedding_var", self.member_embedding_var)
            self.member_batch_embedded = tf.nn.embedding_lookup(self.member_embedding_var, self.member_batch_ph)

            self.campaign_embedding_var = tf.get_variable("campaign_embedding_var", [n_campaign, EMBEDDING_DIM // 3])
            tf.summary.histogram("campaign_embedding_var", self.campaign_embedding_var)
            self.campaign_batch_embedded = tf.nn.embedding_lookup(self.campaign_embedding_var, self.campaign_batch_ph)


            # item embedding * item price
            self.item_embedding_var = tf.get_variable("item_embedding_var", [n_item, EMBEDDING_DIM // 3])
            tf.summary.histogram("item_embedding_var", self.item_embedding_var)
            self.item_batch_embedded_raw = tf.nn.embedding_lookup(self.item_embedding_var, self.item_batch_ph)
            tmp_item_price_batch_ph = tf.expand_dims(self.item_price_batch_ph, -1)
            self.item_batch_embedded = tmp_item_price_batch_ph * self.item_batch_embedded_raw


            self.cate_embedding_var = tf.get_variable("cate_embedding_var", [n_cate, EMBEDDING_DIM // 3])
            tf.summary.histogram("cate_embedding_var", self.cate_embedding_var)
            self.cate_batch_embedded = tf.nn.embedding_lookup(self.cate_embedding_var, self.cate_batch_ph)

            self.commodity_embedding_var = tf.get_variable("commodity_embedding_var", [n_commodity, EMBEDDING_DIM // 3])
            tf.summary.histogram("commodity_embedding_var", self.commodity_embedding_var)
            self.commodity_batch_embedded = tf.nn.embedding_lookup(self.commodity_embedding_var, self.commodity_batch_ph)

            self.node_embedding_var = tf.get_variable("node_embedding_var", [n_node, EMBEDDING_DIM // 3])
            tf.summary.histogram("node_embedding_var", self.node_embedding_var)
            self.node_batch_embedded = tf.nn.embedding_lookup(self.node_embedding_var, self.node_batch_ph)

            # effect data
            self.effect_embedding_var = tf.get_variable("effect_embedding_var", [n_effect, EMBEDDING_DIM // 6])
            tf.summary.histogram("effect_embedding_var", self.effect_embedding_var)
            self.effect_batch_his_embedded_raw = tf.nn.embedding_lookup(self.effect_embedding_var, self.effect_batch_his_id_ph)
            tmp_effect_batch_his_val_ph = tf.expand_dims(self.effect_batch_his_val_ph, -1)
            tmp_effect_batch_his_embedded = tmp_effect_batch_his_val_ph * self.effect_batch_his_embedded_raw
            self.effect_batch_his_embedded = tf.reshape(tmp_effect_batch_his_embedded, [tf.shape(tmp_effect_batch_his_embedded)[0], tf.shape(tmp_effect_batch_his_embedded)[1], n_effect * EMBEDDING_DIM // 6]) # batch * days * (features * embedding dims)

            # pos data
            self.pos_embedding_var = tf.get_variable("pos_embedding_var", [n_pos, EMBEDDING_DIM // 3])
            tf.summary.histogram("pos_embedding_var", self.pos_embedding_var)
            self.pos_batch_his_embedded_raw = tf.nn.embedding_lookup(self.pos_embedding_var, self.pos_batch_his_id_ph)
            tmp_pos_batch_his_val_ph = tf.expand_dims(self.pos_batch_his_val_ph, -1)
            tmp_pos_batch_his_embedded = tmp_pos_batch_his_val_ph * self.pos_batch_his_embedded_raw
            self.pos_batch_his_embedded = tf.reshape(tmp_pos_batch_his_embedded, [tf.shape(tmp_pos_batch_his_embedded)[0], tf.shape(tmp_pos_batch_his_embedded)[1], n_pos * EMBEDDING_DIM // 3]) # batch * days * (features * embedding dims)

            # direct data
            self.direct_embedding_var = tf.get_variable("direct_embedding_var", [n_direct, EMBEDDING_DIM])
            tf.summary.histogram("direct_embedding_var", self.direct_embedding_var)
            self.direct_batch_his_embedded_raw = tf.nn.embedding_lookup(self.direct_embedding_var, self.direct_batch_his_id_ph)
            tmp_direct_batch_his_val_ph = tf.expand_dims(self.direct_batch_his_val_ph, -1)
            self.direct_batch_his_embedded = tmp_direct_batch_his_val_ph * self.direct_batch_his_embedded_raw # batch * days * maxlen * embedding dims

            # direct action
            self.direct_action_embedding_var = tf.get_variable("direct_action_embedding_var", [n_direct_action, EMBEDDING_DIM])
            tf.summary.histogram("direct_action_embedding_var", self.direct_action_embedding_var)
            self.direct_action_batch_his_embedded_raw = tf.nn.embedding_lookup(self.direct_action_embedding_var, self.direct_action_batch_his_id_ph)
            tmp_direct_action_batch_his_val_ph = tf.expand_dims(self.direct_action_batch_his_val_ph, -1)
            self.direct_action_batch_his_embedded = tmp_direct_action_batch_his_val_ph * self.direct_action_batch_his_embedded_raw # batch * days * maxlen * embedding dims

            # 第三维 average pooling 一下，没有操作就是 0

            # pos action
            self.pos_action_embedding_var = tf.get_variable("pos_action_embedding_var", [n_pos_action, EMBEDDING_DIM])
            tf.summary.histogram("pos_action_embedding_var", self.pos_action_embedding_var)
            self.pos_action_batch_his_embedded_raw = tf.nn.embedding_lookup(self.pos_action_embedding_var, self.pos_action_batch_his_id_ph)
            tmp_pos_action_batch_his_val_ph = tf.expand_dims(self.pos_action_batch_his_val_ph, -1)
            self.pos_action_batch_his_embedded = tmp_pos_action_batch_his_val_ph * self.pos_action_batch_his_embedded_raw # batch * days * maxlen * embedding dims



    def debug(self, sess, inps):
        pos_action_type, pos_action_mask, pos_action_val, pos_action_embedding_raw, pos_action_embedding = sess.run([self.pos_action_batch_his_id_ph, \
                self.pos_action_batch_his_mask_ph, \
                self.pos_action_batch_his_val_ph, \
                self.pos_action_batch_his_embedded_raw, \
                self.pos_action_batch_his_embedded], \
            feed_dict={
                self.pos_action_batch_his_id_ph: inps[0],
                self.pos_action_batch_his_val_ph: inps[1],
                self.pos_action_batch_his_mask_ph: inps[2]
            })
        return pos_action_type, pos_action_mask, pos_action_val, pos_action_embedding_raw, pos_action_embedding

    def build_fcn_net(self, inp, use_dice=False):
        # bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        # self.bn1 = bn1
        dnn1 = tf.layers.dense(inp, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def train(self, sess, inps):
        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.lr: inps[0],
                self.y_ph: inps[1],
                self.adgroup_batch_ph: inps[3],
                self.member_batch_ph: inps[4],
                self.campaign_batch_ph: inps[5],
                self.item_batch_ph: inps[6],
                self.item_price_batch_ph: inps[7],
                self.cate_batch_ph: inps[8],
                self.commodity_batch_ph: inps[9],
                self.node_batch_ph: inps[10],
                self.effect_batch_his_id_ph: inps[11],
                self.effect_batch_his_val_ph: inps[12],
                self.effect_batch_his_mask_ph: inps[13],
                self.pos_batch_his_id_ph: inps[14],
                self.pos_batch_his_val_ph: inps[15],
                self.pos_batch_his_mask_ph: inps[16],
                self.direct_batch_his_id_ph: inps[17],
                self.direct_batch_his_val_ph: inps[18],
                self.direct_batch_his_mask_ph: inps[19],
                self.direct_action_batch_his_id_ph: inps[20],
                self.direct_action_batch_his_val_ph: inps[21],
                self.direct_action_batch_his_mask_ph: inps[22],
                self.pos_action_batch_his_id_ph: inps[23],
                self.pos_action_batch_his_val_ph: inps[24],
                self.pos_action_batch_his_mask_ph: inps[25]
            })
        return loss, accuracy

    def calculate(self, sess, inps):
        probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.y_ph: inps[0],
                self.adgroup_batch_ph: inps[2],
                self.member_batch_ph: inps[3],
                self.campaign_batch_ph: inps[4],
                self.item_batch_ph: inps[5],
                self.item_price_batch_ph: inps[6],
                self.cate_batch_ph: inps[7],
                self.commodity_batch_ph: inps[8],
                self.node_batch_ph: inps[9],
                self.effect_batch_his_id_ph: inps[10],
                self.effect_batch_his_val_ph: inps[11],
                self.effect_batch_his_mask_ph: inps[12],
                self.pos_batch_his_id_ph: inps[13],
                self.pos_batch_his_val_ph: inps[14],
                self.pos_batch_his_mask_ph: inps[15],
                self.direct_batch_his_id_ph: inps[16],
                self.direct_batch_his_val_ph: inps[17],
                self.direct_batch_his_mask_ph: inps[18],
                self.direct_action_batch_his_id_ph: inps[19],
                self.direct_action_batch_his_val_ph: inps[20],
                self.direct_action_batch_his_mask_ph: inps[21],
                self.pos_action_batch_his_id_ph: inps[22],
                self.pos_action_batch_his_val_ph: inps[23],
                self.pos_action_batch_his_mask_ph: inps[24]
            })
        return probs, loss, accuracy

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

class Model_MLP(Model):
    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)

        self.ad_eb = tf.concat([ \
                # self.adgroup_batch_embedded, \
                self.member_batch_embedded, \
                # self.campaign_batch_embedded, \
                # self.item_batch_embedded, \
                self.cate_batch_embedded, \
                self.commodity_batch_embedded, \
                self.node_batch_embedded \
                ], -1)

        # direct info: sum pooling
        tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
        self.mlp_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

        # direct action info: average pooling
        tmp_direct_action_batch_his_mask_ph = tf.expand_dims(self.direct_action_batch_his_mask_ph, -1)
        self.mlp_direct_action_batch_his_embedded = tf.reduce_sum(tmp_direct_action_batch_his_mask_ph * self.direct_action_batch_his_embedded, 2)
        self.mlp_direct_action_mask = tf.reduce_sum(self.direct_action_batch_his_mask_ph, 2) + 0.000000001
        tmp_mlp_direct_action_mask = tf.expand_dims(self.mlp_direct_action_mask, -1)
        self.mlp_direct_action_batch_his_embedded = self.mlp_direct_action_batch_his_embedded / tmp_mlp_direct_action_mask

        # pos action info: average pooling
        tmp_pos_action_batch_his_mask_ph = tf.expand_dims(self.pos_action_batch_his_mask_ph, -1)
        self.mlp_pos_action_batch_his_embedded = tf.reduce_sum(tmp_pos_action_batch_his_mask_ph * self.pos_action_batch_his_embedded, 2)
        self.mlp_pos_action_mask = tf.reduce_sum(self.pos_action_batch_his_mask_ph, 2) + 0.000000001
        tmp_mlp_pos_action_mask = tf.expand_dims(self.mlp_pos_action_mask, -1)
        self.mlp_pos_action_batch_his_embedded = self.mlp_pos_action_batch_his_embedded / tmp_mlp_pos_action_mask

        state_action_feature = tf.concat([ \
                self.effect_batch_his_embedded, \
                self.pos_batch_his_embedded, \
                self.mlp_direct_batch_his_embedded, \
                self.mlp_direct_action_batch_his_embedded, \
                self.mlp_pos_action_batch_his_embedded
                ], -1)

        state_action_feature = tf.reduce_sum(state_action_feature, 1)
        inp = tf.concat([self.ad_eb, state_action_feature], -1)
        # inp = tf.concat([self.ad_eb, self.state_action_sum_eb, attention_feature], -1)
        self.build_fcn_net(inp)

class Model_WideDeep(Model):
    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)

        self.ad_eb = tf.concat([ \
                # self.adgroup_batch_embedded, \
                self.member_batch_embedded, \
                # self.campaign_batch_embedded, \
                # self.item_batch_embedded, \
                self.cate_batch_embedded, \
                self.commodity_batch_embedded, \
                self.node_batch_embedded \
                ], -1)

        # direct info: sum pooling
        tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
        self.widedeep_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

        # direct action info: average pooling
        tmp_direct_action_batch_his_mask_ph = tf.expand_dims(self.direct_action_batch_his_mask_ph, -1)
        self.widedeep_direct_action_batch_his_embedded = tf.reduce_sum(tmp_direct_action_batch_his_mask_ph * self.direct_action_batch_his_embedded, 2)
        self.widedeep_direct_action_mask = tf.reduce_sum(self.direct_action_batch_his_mask_ph, 2) + 0.000000001
        tmp_widedeep_direct_action_mask = tf.expand_dims(self.widedeep_direct_action_mask, -1)
        self.widedeep_direct_action_batch_his_embedded = self.widedeep_direct_action_batch_his_embedded / tmp_widedeep_direct_action_mask

        # pos action info: average pooling
        tmp_pos_action_batch_his_mask_ph = tf.expand_dims(self.pos_action_batch_his_mask_ph, -1)
        self.widedeep_pos_action_batch_his_embedded = tf.reduce_sum(tmp_pos_action_batch_his_mask_ph * self.pos_action_batch_his_embedded, 2)
        self.widedeep_pos_action_mask = tf.reduce_sum(self.pos_action_batch_his_mask_ph, 2) + 0.000000001
        tmp_widedeep_pos_action_mask = tf.expand_dims(self.widedeep_pos_action_mask, -1)
        self.widedeep_pos_action_batch_his_embedded = self.widedeep_pos_action_batch_his_embedded / tmp_widedeep_pos_action_mask

        state_action_feature = tf.concat([ \
            self.effect_batch_his_embedded, \
            self.pos_batch_his_embedded, \
            self.widedeep_direct_batch_his_embedded, \
            self.widedeep_direct_action_batch_his_embedded, \
            self.widedeep_pos_action_batch_his_embedded
        ], -1)

        self.state_eb = tf.reduce_sum(state_action_feature, 1)
        inp = tf.concat([self.ad_eb, self.state_eb], -1)
        # Fully connected layer
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'p1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'p2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        d_layer_wide = tf.concat([self.ad_eb, self.state_eb], -1)
        d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='wide')
        self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide)

        with tf.name_scope('Metric'):
            self.target_ph = self.y_ph
            self.loss = -tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))

class Model_PNN(Model):
    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)

        ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

        ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 4, activation=None, name='ad_feature_f1')
        ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
        ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f2')
        ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
        self.ad_eb = ad_dnn2

        # direct info: sum pooling
        tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
        self.pnn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

        # direct action info: average pooling
        tmp_direct_action_batch_his_mask_ph = tf.expand_dims(self.direct_action_batch_his_mask_ph, -1)
        self.pnn_direct_action_batch_his_embedded = tf.reduce_sum(tmp_direct_action_batch_his_mask_ph * self.direct_action_batch_his_embedded, 2)
        self.pnn_direct_action_mask = tf.reduce_sum(self.direct_action_batch_his_mask_ph, 2) + 0.000000001
        tmp_pnn_direct_action_mask = tf.expand_dims(self.pnn_direct_action_mask, -1)
        self.pnn_direct_action_batch_his_embedded = self.pnn_direct_action_batch_his_embedded / tmp_pnn_direct_action_mask

        # pos action info: average pooling
        tmp_pos_action_batch_his_mask_ph = tf.expand_dims(self.pos_action_batch_his_mask_ph, -1)
        self.pnn_pos_action_batch_his_embedded = tf.reduce_sum(tmp_pos_action_batch_his_mask_ph * self.pos_action_batch_his_embedded, 2)
        self.pnn_pos_action_mask = tf.reduce_sum(self.pos_action_batch_his_mask_ph, 2) + 0.000000001
        tmp_pnn_pos_action_mask = tf.expand_dims(self.pnn_pos_action_mask, -1)
        self.pnn_pos_action_batch_his_embedded = self.pnn_pos_action_batch_his_embedded / tmp_pnn_pos_action_mask

        state_action_feature = tf.concat([ \
                self.effect_batch_his_embedded, \
                self.pos_batch_his_embedded, \
                self.pnn_direct_batch_his_embedded, \
                self.pnn_direct_action_batch_his_embedded, \
                self.pnn_pos_action_batch_his_embedded
                ], -1)

        state_action_feature = tf.reduce_sum(state_action_feature, 1)

        state_dnn1 = tf.layers.dense(state_action_feature, EMBEDDING_DIM * 4, activation=None, name='state_feature_f1')
        state_dnn1 = prelu(state_dnn1, 'state_feature_p1')
        state_dnn2 = tf.layers.dense(state_dnn1, EMBEDDING_DIM * 2, activation=None, name='state_feature_f2')
        state_dnn2 = prelu(state_dnn2, 'state_feature_p2')
        self.state_eb = state_dnn2
        # inp = tf.concat([self.ad_eb, self.state_action_sum_eb, attention_feature], -1)
        inp = tf.concat([self.ad_eb, self.state_eb, self.ad_eb * self.state_eb], -1)
        self.build_fcn_net(inp)

class Model_DIN(Model):
    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)

        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 4, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2
            # self.ad_eb = tf.reshape(self.ad_eb, [-1, EMBEDDING_DIM * 2])

        with tf.name_scope('State_action_feature_embedding'):
            
            # direct info: sum pooling
            tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
            self.din_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

            # direct action info: average pooling
            tmp_direct_action_batch_his_mask_ph = tf.expand_dims(self.direct_action_batch_his_mask_ph, -1)
            self.din_direct_action_batch_his_embedded = tf.reduce_sum(tmp_direct_action_batch_his_mask_ph * self.direct_action_batch_his_embedded, 2)
            self.din_direct_action_mask = tf.reduce_sum(self.direct_action_batch_his_mask_ph, 2) + 0.000000001
            tmp_din_direct_action_mask = tf.expand_dims(self.din_direct_action_mask, -1)
            self.din_direct_action_batch_his_embedded = self.din_direct_action_batch_his_embedded / tmp_din_direct_action_mask

            # pos action info: average pooling
            tmp_pos_action_batch_his_mask_ph = tf.expand_dims(self.pos_action_batch_his_mask_ph, -1)
            self.din_pos_action_batch_his_embedded = tf.reduce_sum(tmp_pos_action_batch_his_mask_ph * self.pos_action_batch_his_embedded, 2)
            self.din_pos_action_mask = tf.reduce_sum(self.pos_action_batch_his_mask_ph, 2) + 0.000000001
            tmp_din_pos_action_mask = tf.expand_dims(self.din_pos_action_mask, -1)
            self.din_pos_action_batch_his_embedded = self.din_pos_action_batch_his_embedded / tmp_din_pos_action_mask

            state_action_feature = tf.concat([ \
                    self.effect_batch_his_embedded, \
                    self.pos_batch_his_embedded, \
                    self.din_direct_batch_his_embedded, \
                    self.din_direct_action_batch_his_embedded, \
                    self.din_pos_action_batch_his_embedded
                    ], -1)

            state_action_dnn1 = tf.layers.dense(state_action_feature, EMBEDDING_DIM * 4, activation=None, name='state_action_f1')
            state_action_dnn1 = prelu(state_action_dnn1, 'state_action_p1')
            state_action_dnn2 = tf.layers.dense(state_action_dnn1, EMBEDDING_DIM * 2, activation=None, name='state_action_f2')
            state_action_dnn2 = prelu(state_action_dnn2, 'state_action_p2')
            self.state_action_eb = state_action_dnn2

            # self.state_action_sum_eb = tf.reduce_sum(self.state_action_eb, -2)
            # self.state_action_sum_eb = tf.reshape(self.state_action_sum_eb, [-1, EMBEDDING_DIM * 2])

        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.ad_eb, self.state_action_eb, ATTENTION_SIZE, self.effect_batch_his_mask_ph)
            attention_feature = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('attention_feature', attention_feature)
            # attention_feature = tf.reshape(attention_feature, [-1, EMBEDDING_DIM * 2])
        inp = tf.concat([self.ad_eb, attention_feature], -1)
        # inp = tf.concat([self.ad_eb, self.state_action_sum_eb, attention_feature], -1)
        self.build_fcn_net(inp, use_dice=True)

class Model_RNN(Model):
    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)

        ad_feature = tf.concat([ \
                # self.adgroup_batch_embedded, \
                self.member_batch_embedded, \
                # self.campaign_batch_embedded, \
                # self.item_batch_embedded, \
                self.cate_batch_embedded, \
                self.commodity_batch_embedded, \
                self.node_batch_embedded \
                ], -1)

        ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 4, activation=None, name='ad_feature_f1')
        ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
        ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f2')
        ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
        self.ad_eb = ad_dnn2

        # direct info: sum pooling
        tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
        self.mlp_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

        # direct action info: average pooling
        tmp_direct_action_batch_his_mask_ph = tf.expand_dims(self.direct_action_batch_his_mask_ph, -1)
        self.mlp_direct_action_batch_his_embedded = tf.reduce_sum(tmp_direct_action_batch_his_mask_ph * self.direct_action_batch_his_embedded, 2)
        self.mlp_direct_action_mask = tf.reduce_sum(self.direct_action_batch_his_mask_ph, 2) + 0.000000001
        tmp_mlp_direct_action_mask = tf.expand_dims(self.mlp_direct_action_mask, -1)
        self.mlp_direct_action_batch_his_embedded = self.mlp_direct_action_batch_his_embedded / tmp_mlp_direct_action_mask

        # pos action info: average pooling
        tmp_pos_action_batch_his_mask_ph = tf.expand_dims(self.pos_action_batch_his_mask_ph, -1)
        self.mlp_pos_action_batch_his_embedded = tf.reduce_sum(tmp_pos_action_batch_his_mask_ph * self.pos_action_batch_his_embedded, 2)
        self.mlp_pos_action_mask = tf.reduce_sum(self.pos_action_batch_his_mask_ph, 2) + 0.000000001
        tmp_mlp_pos_action_mask = tf.expand_dims(self.mlp_pos_action_mask, -1)
        self.mlp_pos_action_batch_his_embedded = self.mlp_pos_action_batch_his_embedded / tmp_mlp_pos_action_mask

        state_action_feature = tf.concat([ \
                self.effect_batch_his_embedded, \
                self.pos_batch_his_embedded, \
                self.mlp_direct_batch_his_embedded, \
                self.mlp_direct_action_batch_his_embedded, \
                self.mlp_pos_action_batch_his_embedded
                ], -1)

        self.ad_eb = tf.tile(self.ad_eb, [1, tf.shape(state_action_feature)[1]])
        self.ad_eb = tf.reshape(self.ad_eb, [tf.shape(self.ad_eb)[0], -1, EMBEDDING_DIM * 2])
        inp = tf.concat([self.ad_eb, state_action_feature], -1)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(GRUCell(EMBEDDING_DIM), inputs=inp, dtype=tf.float32, scope='gru')
        self.y_hat = tf.nn.sigmoid(tf.reduce_sum(rnn_states, -1))
        self.y_hat = tf.reshape(self.y_hat, [-1, 1])
        self.y_hat = tf.concat([self.y_hat, 1.0 - self.y_hat], -1)

        # Cross-entropy loss and optimizer initialization
        self.target_ph = self.y_ph
        loss = -tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
        self.loss = loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # Accuracy metric
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))

class Model_biRNN(Model):
    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)

        ad_feature = tf.concat([ \
                # self.adgroup_batch_embedded, \
                self.member_batch_embedded, \
                # self.campaign_batch_embedded, \
                # self.item_batch_embedded, \
                self.cate_batch_embedded, \
                self.commodity_batch_embedded, \
                self.node_batch_embedded \
                ], -1)

        ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 4, activation=None, name='ad_feature_f1')
        ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
        ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f2')
        ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
        self.ad_eb = ad_dnn2

        # direct info: sum pooling
        tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
        self.mlp_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

        # direct action info: average pooling
        tmp_direct_action_batch_his_mask_ph = tf.expand_dims(self.direct_action_batch_his_mask_ph, -1)
        self.mlp_direct_action_batch_his_embedded = tf.reduce_sum(tmp_direct_action_batch_his_mask_ph * self.direct_action_batch_his_embedded, 2)
        self.mlp_direct_action_mask = tf.reduce_sum(self.direct_action_batch_his_mask_ph, 2) + 0.000000001
        tmp_mlp_direct_action_mask = tf.expand_dims(self.mlp_direct_action_mask, -1)
        self.mlp_direct_action_batch_his_embedded = self.mlp_direct_action_batch_his_embedded / tmp_mlp_direct_action_mask

        # pos action info: average pooling
        tmp_pos_action_batch_his_mask_ph = tf.expand_dims(self.pos_action_batch_his_mask_ph, -1)
        self.mlp_pos_action_batch_his_embedded = tf.reduce_sum(tmp_pos_action_batch_his_mask_ph * self.pos_action_batch_his_embedded, 2)
        self.mlp_pos_action_mask = tf.reduce_sum(self.pos_action_batch_his_mask_ph, 2) + 0.000000001
        tmp_mlp_pos_action_mask = tf.expand_dims(self.mlp_pos_action_mask, -1)
        self.mlp_pos_action_batch_his_embedded = self.mlp_pos_action_batch_his_embedded / tmp_mlp_pos_action_mask

        state_action_feature = tf.concat([ \
                self.effect_batch_his_embedded, \
                self.pos_batch_his_embedded, \
                self.mlp_direct_batch_his_embedded, \
                self.mlp_direct_action_batch_his_embedded, \
                self.mlp_pos_action_batch_his_embedded
                ], -1)

        self.ad_eb = tf.tile(self.ad_eb, [1, tf.shape(state_action_feature)[1]])
        self.ad_eb = tf.reshape(self.ad_eb, [tf.shape(self.ad_eb)[0], -1, EMBEDDING_DIM * 2])
        inp = tf.concat([self.ad_eb, state_action_feature], -1)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(GRUCell(EMBEDDING_DIM), GRUCell(EMBEDDING_DIM), inputs=inp, dtype=tf.float32, scope='gru')
        self.y_hat = tf.nn.sigmoid(tf.reduce_sum(rnn_states[0], -1) + tf.reduce_sum(rnn_states[1], -1))
        self.y_hat = tf.reshape(self.y_hat, [-1, 1])
        self.y_hat = tf.concat([self.y_hat, 1.0 - self.y_hat], -1)

        # Cross-entropy loss and optimizer initialization
        self.target_ph = self.y_ph
        loss = -tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
        self.loss = loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # Accuracy metric
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))

class Model_DSPN(Model):
    def get_w(self, sess, inps):
        www = sess.run(self.www, feed_dict={
                self.y_ph: inps[0],
                self.adgroup_batch_ph: inps[2],
                self.member_batch_ph: inps[3],
                self.campaign_batch_ph: inps[4],
                self.item_batch_ph: inps[5],
                self.item_price_batch_ph: inps[6],
                self.cate_batch_ph: inps[7],
                self.commodity_batch_ph: inps[8],
                self.node_batch_ph: inps[9],
                self.effect_batch_his_id_ph: inps[10],
                self.effect_batch_his_val_ph: inps[11],
                self.effect_batch_his_mask_ph: inps[12],
                self.pos_batch_his_id_ph: inps[13],
                self.pos_batch_his_val_ph: inps[14],
                self.pos_batch_his_mask_ph: inps[15],
                self.direct_batch_his_id_ph: inps[16],
                self.direct_batch_his_val_ph: inps[17],
                self.direct_batch_his_mask_ph: inps[18],
                self.direct_action_batch_his_id_ph: inps[19],
                self.direct_action_batch_his_val_ph: inps[20],
                self.direct_action_batch_his_mask_ph: inps[21],
                self.pos_action_batch_his_id_ph: inps[22],
                self.pos_action_batch_his_val_ph: inps[23],
                self.pos_action_batch_his_mask_ph: inps[24]
            })
        return www

    def DAVN_att_self_attention(self, facts, mask):

        # facts: batch * day * maxlen * embedding
        # mask: batch * day * maxlen
        tmp_mask = tf.expand_dims(mask, -1)
        tmp_mask = tf.matmul(tmp_mask, tf.transpose(tmp_mask, [0, 1, 3, 2]))
        key_mask = tf.equal(tmp_mask, tf.ones_like(tmp_mask))
        tmp_mat = tf.matmul(facts, tf.transpose(facts, [0, 1, 3, 2]))
        paddings = tf.ones_like(tmp_mat) * (-2 ** 32 + 1)
        tmp_mat = tf.where(key_mask, tmp_mat, paddings)
        tmp_mat = tf.nn.softmax(tmp_mat, name='alphas')
        output = tf.matmul(tmp_mat, facts)
        output = output * tf.expand_dims(mask, -1)
        return output

    def DAVN_att_ad_attention(self, query, facts, mask, EMBEDDING_DIM, type='null'):
        
        # facts: batch * day * maxlen * embedding
        # query: batch * embedding
        query = tf.tile(query, [1, tf.shape(facts)[1]])
        query = tf.reshape(query, [tf.shape(facts)[0], tf.shape(facts)[1], EMBEDDING_DIM])

        # direct info: sum pooling
        tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
        self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)
        query = tf.concat([self.effect_batch_his_embedded, self.davn_direct_batch_his_embedded, self.pos_batch_his_embedded, query], -1)
        with tf.name_scope('DAVN_att_ad_attention_ad_feature_embedding'):
            ad_att_dnn1 = tf.layers.dense(query, EMBEDDING_DIM * 2, activation=None)
            ad_att_dnn1 = prelu(ad_att_dnn1, 'ad_att_dnn1_' + type)
            ad_att_dnn2 = tf.layers.dense(ad_att_dnn1, EMBEDDING_DIM, activation=None)
            ad_att_dnn2 = prelu(ad_att_dnn2, 'ad_att_dnn2_' + type)
        query = ad_att_dnn2

        tmp_weight = tf.matmul(facts, tf.expand_dims(query, -1)) # batch * day * maxlen * 1
        tmp_weight = tf.reshape(tmp_weight, [tf.shape(tmp_weight)[0], \
                        tf.shape(tmp_weight)[1], tf.shape(tmp_weight)[2]])
        key_mask = tf.equal(mask, tf.ones_like(mask))
        paddings = tf.ones_like(tmp_weight) * (-2 * 32 + 1)
        tmp_weight = tf.where(key_mask, tmp_weight, paddings)
        tmp_weight = tf.nn.softmax(tmp_weight, name='alphas')
        output = facts * tf.expand_dims(tmp_weight, -1)
        output = tf.reduce_sum(output, 2)
        return output


    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)
        
        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2

        with tf.name_scope('direct_actions'):
            self.direct_action_his_info = self.DAVN_att_self_attention(self.direct_action_batch_his_embedded, self.direct_action_batch_his_mask_ph)
            self.direct_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.direct_action_his_info, self.direct_action_batch_his_mask_ph, EMBEDDING_DIM, type='direct')

        with tf.name_scope('pos_actions'):
            self.pos_action_his_info = self.DAVN_att_self_attention(self.pos_action_batch_his_embedded, self.pos_action_batch_his_mask_ph)
            self.pos_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.pos_action_his_info, self.pos_action_batch_his_mask_ph, EMBEDDING_DIM, type='pos')

        with tf.name_scope('rnn'):

            self.action_info = tf.concat([self.direct_action_info, self.pos_action_info], -1)
            self.action_info = tf.reshape(self.action_info, [tf.shape(self.action_info)[0], tf.shape(self.action_info)[1], EMBEDDING_DIM * 2])

            # direct info: sum pooling
            tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
            self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

            self.state_action_info = tf.concat([self.effect_batch_his_embedded, self.pos_batch_his_embedded, self.davn_direct_batch_his_embedded, self.action_info], -1)

            self.state_action_info_bn = tf.layers.batch_normalization(inputs=self.state_action_info, name='state_action_info_bn')
            
            bi_rnn_outputs1, bi_rnn_states1 = tf.nn.bidirectional_dynamic_rnn(GRUCell(EMBEDDING_DIM), GRUCell(EMBEDDING_DIM), inputs=self.state_action_info_bn, \
                                       dtype=tf.float32, scope='gru1')
            bi_rnn_inputs2 = tf.concat([bi_rnn_outputs1[0], bi_rnn_outputs1[1]], -1)
            rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(GRUCell(n_effect + 1), GRUCell(n_effect + 1), inputs=bi_rnn_inputs2, \
                                       dtype=tf.float32, scope='gru2')
            
            self.intent = rnn_states2[0][:, 0 : n_effect] + rnn_states2[1][ :, 0 : n_effect]
            self.bias = rnn_states2[0][ :, -1] + rnn_states2[1][ :, -1]
            self.bias = tf.reshape(self.bias, [-1, 1])

            self.www = rnn_states2[0] + rnn_states2[1]
        
        with tf.name_scope('satisfaction'):
            self.intent = tf.tile(self.intent, [1, tf.shape(self.effect_batch_his_val_ph)[1]]) #
            self.intent = tf.reshape(self.intent, [tf.shape(self.effect_batch_his_val_ph)[0], tf.shape(self.effect_batch_his_val_ph)[1], tf.shape(self.effect_batch_his_val_ph)[2]]) 
            # self.effect_batch_his_val_ph: batch * day * effect
            self.satisfaction = tf.matmul(tf.expand_dims(self.effect_batch_his_val_ph, -2), tf.expand_dims(self.intent, -1))
            self.satisfaction = tf.reshape(self.satisfaction, [tf.shape(self.satisfaction)[0], tf.shape(self.satisfaction)[1]])
            self.satisfaction = self.satisfaction + self.bias
            self.satisfaction = tf.sigmoid(self.satisfaction) / tf.to_float(tf.shape(self.satisfaction)[1])
            self.satisfaction = tf.reduce_sum(self.satisfaction, -1)
            self.satisfaction = tf.reshape(self.satisfaction, [-1, 1])
            self.satisfaction = self.satisfaction
            self.y_hat = tf.concat([self.satisfaction, 1 - self.satisfaction], -1)
            # self.probability = tf.nn.sigmoid(self.probability)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-9, 1.0)) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

class Model_DSPN_RNN(Model):
    def DAVN_att_self_attention(self, facts, mask):

        # facts: batch * day * maxlen * embedding
        # mask: batch * day * maxlen
        tmp_mask = tf.expand_dims(mask, -1)
        tmp_mask = tf.matmul(tmp_mask, tf.transpose(tmp_mask, [0, 1, 3, 2]))
        key_mask = tf.equal(tmp_mask, tf.ones_like(tmp_mask))
        tmp_mat = tf.matmul(facts, tf.transpose(facts, [0, 1, 3, 2]))
        paddings = tf.ones_like(tmp_mat) * (-2 ** 32 + 1)
        tmp_mat = tf.where(key_mask, tmp_mat, paddings)
        tmp_mat = tf.nn.softmax(tmp_mat, name='alphas')
        output = tf.matmul(tmp_mat, facts)
        output = output * tf.expand_dims(mask, -1)
        return output

    def DAVN_att_ad_attention(self, query, facts, mask, EMBEDDING_DIM, type='null'):
        
        # facts: batch * day * maxlen * embedding
        # query: batch * embedding
        query = tf.tile(query, [1, tf.shape(facts)[1]])
        query = tf.reshape(query, [tf.shape(facts)[0], tf.shape(facts)[1], EMBEDDING_DIM])

        # direct info: sum pooling
        tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
        self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)
        query = tf.concat([self.effect_batch_his_embedded, self.davn_direct_batch_his_embedded, self.pos_batch_his_embedded, query], -1)
        with tf.name_scope('DAVN_att_ad_attention_ad_feature_embedding'):
            ad_att_dnn1 = tf.layers.dense(query, EMBEDDING_DIM * 2, activation=None)
            ad_att_dnn1 = prelu(ad_att_dnn1, 'ad_att_dnn1_' + type)
            ad_att_dnn2 = tf.layers.dense(ad_att_dnn1, EMBEDDING_DIM, activation=None)
            ad_att_dnn2 = prelu(ad_att_dnn2, 'ad_att_dnn2_' + type)
        query = ad_att_dnn2

        tmp_weight = tf.matmul(facts, tf.expand_dims(query, -1)) # batch * day * maxlen * 1
        tmp_weight = tf.reshape(tmp_weight, [tf.shape(tmp_weight)[0], \
                        tf.shape(tmp_weight)[1], tf.shape(tmp_weight)[2]])
        key_mask = tf.equal(mask, tf.ones_like(mask))
        paddings = tf.ones_like(tmp_weight) * (-2 * 32 + 1)
        tmp_weight = tf.where(key_mask, tmp_weight, paddings)
        tmp_weight = tf.nn.softmax(tmp_weight, name='alphas')
        output = facts * tf.expand_dims(tmp_weight, -1)
        output = tf.reduce_sum(output, 2)
        return output

    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)
        
        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2

        with tf.name_scope('direct_actions'):
            self.direct_action_his_info = self.DAVN_att_self_attention(self.direct_action_batch_his_embedded, self.direct_action_batch_his_mask_ph)
            self.direct_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.direct_action_his_info, self.direct_action_batch_his_mask_ph, EMBEDDING_DIM, type='direct')

        with tf.name_scope('pos_actions'):
            self.pos_action_his_info = self.DAVN_att_self_attention(self.pos_action_batch_his_embedded, self.pos_action_batch_his_mask_ph)
            self.pos_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.pos_action_his_info, self.pos_action_batch_his_mask_ph, EMBEDDING_DIM, type='pos')

        with tf.name_scope('rnn'):

            self.action_info = tf.concat([self.direct_action_info, self.pos_action_info], -1)
            self.action_info = tf.reshape(self.action_info, [tf.shape(self.action_info)[0], tf.shape(self.action_info)[1], EMBEDDING_DIM * 2])

            # direct info: sum pooling
            tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
            self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

            self.state_action_info = tf.concat([self.effect_batch_his_embedded, self.pos_batch_his_embedded, self.davn_direct_batch_his_embedded, self.action_info], -1)

            self.state_action_info_bn = tf.layers.batch_normalization(inputs=self.state_action_info, name='state_action_info_bn')
            rnn_outputs1, rnn_states1 = tf.nn.dynamic_rnn(GRUCell(EMBEDDING_DIM), inputs=self.state_action_info_bn, \
                                       dtype=tf.float32, scope='gru1')
            rnn_outputs2, rnn_states2 = tf.nn.dynamic_rnn(GRUCell(n_effect + 1), inputs=rnn_outputs1, \
                                       dtype=tf.float32, scope='gru2')
            
            self.intent = rnn_states2[ :, 0 : n_effect]
            self.bias = rnn_states2[ :, -1]
            self.bias = tf.reshape(self.bias, [-1, 1])
        
        with tf.name_scope('satisfaction'):
            self.intent = tf.tile(self.intent, [1, tf.shape(self.effect_batch_his_val_ph)[1]]) #
            self.intent = tf.reshape(self.intent, [tf.shape(self.effect_batch_his_val_ph)[0], tf.shape(self.effect_batch_his_val_ph)[1], tf.shape(self.effect_batch_his_val_ph)[2]]) 
            # self.effect_batch_his_val_ph: batch * day * effect
            self.satisfaction = tf.matmul(tf.expand_dims(self.effect_batch_his_val_ph, -2), tf.expand_dims(self.intent, -1))
            self.satisfaction = tf.reshape(self.satisfaction, [tf.shape(self.satisfaction)[0], tf.shape(self.satisfaction)[1]])
            self.satisfaction = self.satisfaction + self.bias
            self.satisfaction = tf.sigmoid(self.satisfaction) / tf.to_float(tf.shape(self.satisfaction)[1])
            self.satisfaction = tf.reduce_sum(self.satisfaction, -1)
            self.satisfaction = tf.reshape(self.satisfaction, [-1, 1])
            self.satisfaction = self.satisfaction
            self.y_hat = tf.concat([self.satisfaction, 1 - self.satisfaction], -1)
            # self.probability = tf.nn.sigmoid(self.probability)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-9, 1.0)) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

class Model_DSPN_MLP(Model):
    def DAVN_att_self_attention(self, facts, mask):

        # facts: batch * day * maxlen * embedding
        # mask: batch * day * maxlen
        tmp_mask = tf.expand_dims(mask, -1)
        tmp_mask = tf.matmul(tmp_mask, tf.transpose(tmp_mask, [0, 1, 3, 2]))
        key_mask = tf.equal(tmp_mask, tf.ones_like(tmp_mask))
        tmp_mat = tf.matmul(facts, tf.transpose(facts, [0, 1, 3, 2]))
        paddings = tf.ones_like(tmp_mat) * (-2 ** 32 + 1)
        tmp_mat = tf.where(key_mask, tmp_mat, paddings)
        tmp_mat = tf.nn.softmax(tmp_mat, name='alphas')
        output = tf.matmul(tmp_mat, facts)
        output = output * tf.expand_dims(mask, -1)
        return output

    def DAVN_att_ad_attention(self, query, facts, mask, EMBEDDING_DIM, type='null'):
        
        # facts: batch * day * maxlen * embedding
        # query: batch * embedding
        query = tf.tile(query, [1, tf.shape(facts)[1]])
        query = tf.reshape(query, [tf.shape(facts)[0], tf.shape(facts)[1], EMBEDDING_DIM])

        # direct info: sum pooling
        tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
        self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)
        query = tf.concat([self.effect_batch_his_embedded, self.davn_direct_batch_his_embedded, self.pos_batch_his_embedded, query], -1)
        with tf.name_scope('DAVN_att_ad_attention_ad_feature_embedding'):
            ad_att_dnn1 = tf.layers.dense(query, EMBEDDING_DIM * 2, activation=None)
            ad_att_dnn1 = prelu(ad_att_dnn1, 'ad_att_dnn1_' + type)
            ad_att_dnn2 = tf.layers.dense(ad_att_dnn1, EMBEDDING_DIM, activation=None)
            ad_att_dnn2 = prelu(ad_att_dnn2, 'ad_att_dnn2_' + type)
        query = ad_att_dnn2

        tmp_weight = tf.matmul(facts, tf.expand_dims(query, -1)) # batch * day * maxlen * 1
        tmp_weight = tf.reshape(tmp_weight, [tf.shape(tmp_weight)[0], \
                        tf.shape(tmp_weight)[1], tf.shape(tmp_weight)[2]])
        key_mask = tf.equal(mask, tf.ones_like(mask))
        paddings = tf.ones_like(tmp_weight) * (-2 * 32 + 1)
        tmp_weight = tf.where(key_mask, tmp_weight, paddings)
        tmp_weight = tf.nn.softmax(tmp_weight, name='alphas')
        output = facts * tf.expand_dims(tmp_weight, -1)
        output = tf.reduce_sum(output, 2)
        return output


    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)
        
        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2

        with tf.name_scope('direct_actions'):
            self.direct_action_his_info = self.DAVN_att_self_attention(self.direct_action_batch_his_embedded, self.direct_action_batch_his_mask_ph)
            self.direct_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.direct_action_his_info, self.direct_action_batch_his_mask_ph, EMBEDDING_DIM, type='direct')

        with tf.name_scope('pos_actions'):
            self.pos_action_his_info = self.DAVN_att_self_attention(self.pos_action_batch_his_embedded, self.pos_action_batch_his_mask_ph)
            self.pos_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.pos_action_his_info, self.pos_action_batch_his_mask_ph, EMBEDDING_DIM, type='pos')

        with tf.name_scope('rnn'):

            self.action_info = tf.concat([self.direct_action_info, self.pos_action_info], -1)
            self.action_info = tf.reshape(self.action_info, [tf.shape(self.action_info)[0], tf.shape(self.action_info)[1], EMBEDDING_DIM * 2])

            # direct info: sum pooling
            tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
            self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

            self.state_action_info = tf.concat([self.effect_batch_his_embedded, self.pos_batch_his_embedded, self.davn_direct_batch_his_embedded, self.action_info], -1)
            self.state_action_info = tf.reduce_sum(self.state_action_info, 1)
            bn1 = tf.layers.batch_normalization(inputs=self.state_action_info, name='bn1')
            # self.bn1 = bn1
            dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
            dnn1 = prelu(dnn1, 'prelu1')
            dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
            dnn2 = prelu(dnn2, 'prelu2')
            dnn3 = tf.layers.dense(dnn2, n_effect + 1, activation=None, name='f3')
            dnn3 = prelu(dnn3, 'prelu3')
            self.intent = dnn3[ :, 0 : n_effect]
            self.bias = dnn3[ :, -1]
            self.bias = tf.reshape(self.bias, [-1, 1])
        
        with tf.name_scope('satisfaction'):
            self.intent = tf.tile(self.intent, [1, tf.shape(self.effect_batch_his_val_ph)[1]]) #
            self.intent = tf.reshape(self.intent, [tf.shape(self.effect_batch_his_val_ph)[0], tf.shape(self.effect_batch_his_val_ph)[1], tf.shape(self.effect_batch_his_val_ph)[2]]) 
            # self.effect_batch_his_val_ph: batch * day * effect
            self.satisfaction = tf.matmul(tf.expand_dims(self.effect_batch_his_val_ph, -2), tf.expand_dims(self.intent, -1))
            self.satisfaction = tf.reshape(self.satisfaction, [tf.shape(self.satisfaction)[0], tf.shape(self.satisfaction)[1]])
            self.satisfaction = self.satisfaction + self.bias
            self.satisfaction = tf.sigmoid(self.satisfaction) / tf.to_float(tf.shape(self.satisfaction)[1])
            self.satisfaction = tf.reduce_sum(self.satisfaction, -1)
            self.satisfaction = tf.reshape(self.satisfaction, [-1, 1])
            self.satisfaction = self.satisfaction
            self.y_hat = tf.concat([self.satisfaction, 1 - self.satisfaction], -1)
            # self.probability = tf.nn.sigmoid(self.probability)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-9, 1.0)) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

class Model_DSPN_no_att(Model):
    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)
        
        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2
            self.ad_eb = tf.tile(self.ad_eb, [1, tf.shape(self.effect_batch_his_embedded)[1]])
            self.ad_eb = tf.reshape(self.ad_eb, [tf.shape(self.ad_eb)[0], -1, EMBEDDING_DIM])

        with tf.name_scope('rnn'):
            # direct action info: sum pooling
            tmp_direct_action_batch_his_mask_ph = tf.expand_dims(self.direct_action_batch_his_mask_ph, -1)
            self.dspn_no_att_direct_action_batch_his_embedded = tf.reduce_sum(tmp_direct_action_batch_his_mask_ph * self.direct_action_batch_his_embedded, 2)

            # pos action info: sum pooling
            tmp_pos_action_batch_his_mask_ph = tf.expand_dims(self.pos_action_batch_his_mask_ph, -1)
            self.dspn_no_att_pos_action_batch_his_embedded = tf.reduce_sum(tmp_pos_action_batch_his_mask_ph * self.pos_action_batch_his_embedded, 2)

            # direct info: sum pooling
            tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
            self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

            self.state_action_info = tf.concat([self.ad_eb, self.effect_batch_his_embedded, self.pos_batch_his_embedded, self.davn_direct_batch_his_embedded, self.dspn_no_att_direct_action_batch_his_embedded, self.dspn_no_att_pos_action_batch_his_embedded], -1)

            self.state_action_info_bn = tf.layers.batch_normalization(inputs=self.state_action_info, name='state_action_info_bn')
            
            bi_rnn_outputs1, bi_rnn_states1 = tf.nn.bidirectional_dynamic_rnn(GRUCell(EMBEDDING_DIM), GRUCell(EMBEDDING_DIM), inputs=self.state_action_info_bn, \
                                       dtype=tf.float32, scope='gru1')
            bi_rnn_inputs2 = tf.concat([bi_rnn_outputs1[0], bi_rnn_outputs1[1]], -1)
            rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(GRUCell(n_effect + 1), GRUCell(n_effect + 1), inputs=bi_rnn_inputs2, \
                                       dtype=tf.float32, scope='gru2')
            
            self.intent = rnn_states2[0][:, 0 : n_effect] + rnn_states2[1][ :, 0 : n_effect]
            self.bias = rnn_states2[0][ :, -1] + rnn_states2[1][ :, -1]
            self.bias = tf.reshape(self.bias, [-1, 1])
        
        with tf.name_scope('satisfaction'):
            self.intent = tf.tile(self.intent, [1, tf.shape(self.effect_batch_his_val_ph)[1]]) #
            self.intent = tf.reshape(self.intent, [tf.shape(self.effect_batch_his_val_ph)[0], tf.shape(self.effect_batch_his_val_ph)[1], tf.shape(self.effect_batch_his_val_ph)[2]]) 
            # self.effect_batch_his_val_ph: batch * day * effect
            self.satisfaction = tf.matmul(tf.expand_dims(self.effect_batch_his_val_ph, -2), tf.expand_dims(self.intent, -1))
            self.satisfaction = tf.reshape(self.satisfaction, [tf.shape(self.satisfaction)[0], tf.shape(self.satisfaction)[1]])
            self.satisfaction = self.satisfaction + self.bias
            self.satisfaction = tf.sigmoid(self.satisfaction) / tf.to_float(tf.shape(self.satisfaction)[1])
            self.satisfaction = tf.reduce_sum(self.satisfaction, -1)
            self.satisfaction = tf.reshape(self.satisfaction, [-1, 1])
            self.satisfaction = self.satisfaction
            self.y_hat = tf.concat([self.satisfaction, 1 - self.satisfaction], -1)
            # self.probability = tf.nn.sigmoid(self.probability)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-9, 1.0)) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

class Model_DSPN_noID(Model):
    def DAVN_att_self_attention(self, facts, mask):

        # facts: batch * day * maxlen * embedding
        # mask: batch * day * maxlen
        tmp_mask = tf.expand_dims(mask, -1)
        tmp_mask = tf.matmul(tmp_mask, tf.transpose(tmp_mask, [0, 1, 3, 2]))
        key_mask = tf.equal(tmp_mask, tf.ones_like(tmp_mask))
        tmp_mat = tf.matmul(facts, tf.transpose(facts, [0, 1, 3, 2]))
        paddings = tf.ones_like(tmp_mat) * (-2 ** 32 + 1)
        tmp_mat = tf.where(key_mask, tmp_mat, paddings)
        tmp_mat = tf.nn.softmax(tmp_mat, name='alphas')
        output = tf.matmul(tmp_mat, facts)
        output = output * tf.expand_dims(mask, -1)
        return output

    def DAVN_att_ad_attention(self, query, facts, mask, EMBEDDING_DIM, type='null'):
        
        # facts: batch * day * maxlen * embedding
        # query: batch * embedding
        query = tf.tile(query, [1, tf.shape(facts)[1]])
        query = tf.reshape(query, [tf.shape(facts)[0], tf.shape(facts)[1], EMBEDDING_DIM])

        # direct info: sum pooling
        tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
        self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)
        query = tf.concat([self.effect_batch_his_embedded, self.davn_direct_batch_his_embedded, self.pos_batch_his_embedded, query], -1)
        with tf.name_scope('DAVN_att_ad_attention_ad_feature_embedding'):
            ad_att_dnn1 = tf.layers.dense(query, EMBEDDING_DIM * 2, activation=None)
            ad_att_dnn1 = prelu(ad_att_dnn1, 'ad_att_dnn1_' + type)
            ad_att_dnn2 = tf.layers.dense(ad_att_dnn1, EMBEDDING_DIM, activation=None)
            ad_att_dnn2 = prelu(ad_att_dnn2, 'ad_att_dnn2_' + type)
        query = ad_att_dnn2

        tmp_weight = tf.matmul(facts, tf.expand_dims(query, -1)) # batch * day * maxlen * 1
        tmp_weight = tf.reshape(tmp_weight, [tf.shape(tmp_weight)[0], \
                        tf.shape(tmp_weight)[1], tf.shape(tmp_weight)[2]])
        key_mask = tf.equal(mask, tf.ones_like(mask))
        paddings = tf.ones_like(tmp_weight) * (-2 * 32 + 1)
        tmp_weight = tf.where(key_mask, tmp_weight, paddings)
        tmp_weight = tf.nn.softmax(tmp_weight, name='alphas')
        output = facts * tf.expand_dims(tmp_weight, -1)
        output = tf.reduce_sum(output, 2)
        return output


    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)
        
        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2
            self.ad_eb = tf.ones_like(self.ad_eb) * (2 ** 32 - 1)

        with tf.name_scope('direct_actions'):
            self.direct_action_his_info = self.DAVN_att_self_attention(self.direct_action_batch_his_embedded, self.direct_action_batch_his_mask_ph)
            self.direct_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.direct_action_his_info, self.direct_action_batch_his_mask_ph, EMBEDDING_DIM, type='direct')

        with tf.name_scope('pos_actions'):
            self.pos_action_his_info = self.DAVN_att_self_attention(self.pos_action_batch_his_embedded, self.pos_action_batch_his_mask_ph)
            self.pos_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.pos_action_his_info, self.pos_action_batch_his_mask_ph, EMBEDDING_DIM, type='pos')

        with tf.name_scope('rnn'):

            self.action_info = tf.concat([self.direct_action_info, self.pos_action_info], -1)
            self.action_info = tf.reshape(self.action_info, [tf.shape(self.action_info)[0], tf.shape(self.action_info)[1], EMBEDDING_DIM * 2])

            # direct info: sum pooling
            tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
            self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

            self.state_action_info = tf.concat([self.effect_batch_his_embedded, self.pos_batch_his_embedded, self.davn_direct_batch_his_embedded, self.action_info], -1)

            self.state_action_info_bn = tf.layers.batch_normalization(inputs=self.state_action_info, name='state_action_info_bn')
            
            bi_rnn_outputs1, bi_rnn_states1 = tf.nn.bidirectional_dynamic_rnn(GRUCell(EMBEDDING_DIM), GRUCell(EMBEDDING_DIM), inputs=self.state_action_info_bn, \
                                       dtype=tf.float32, scope='gru1')
            bi_rnn_inputs2 = tf.concat([bi_rnn_outputs1[0], bi_rnn_outputs1[1]], -1)
            rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(GRUCell(n_effect + 1), GRUCell(n_effect + 1), inputs=bi_rnn_inputs2, \
                                       dtype=tf.float32, scope='gru2')
            
            self.intent = rnn_states2[0][:, 0 : n_effect] + rnn_states2[1][ :, 0 : n_effect]
            self.bias = rnn_states2[0][ :, -1] + rnn_states2[1][ :, -1]
            self.bias = tf.reshape(self.bias, [-1, 1])
        
        with tf.name_scope('satisfaction'):
            self.intent = tf.tile(self.intent, [1, tf.shape(self.effect_batch_his_val_ph)[1]]) #
            self.intent = tf.reshape(self.intent, [tf.shape(self.effect_batch_his_val_ph)[0], tf.shape(self.effect_batch_his_val_ph)[1], tf.shape(self.effect_batch_his_val_ph)[2]]) 
            # self.effect_batch_his_val_ph: batch * day * effect
            self.satisfaction = tf.matmul(tf.expand_dims(self.effect_batch_his_val_ph, -2), tf.expand_dims(self.intent, -1))
            self.satisfaction = tf.reshape(self.satisfaction, [tf.shape(self.satisfaction)[0], tf.shape(self.satisfaction)[1]])
            self.satisfaction = self.satisfaction + self.bias
            self.satisfaction = tf.sigmoid(self.satisfaction) / tf.to_float(tf.shape(self.satisfaction)[1])
            self.satisfaction = tf.reduce_sum(self.satisfaction, -1)
            self.satisfaction = tf.reshape(self.satisfaction, [-1, 1])
            self.satisfaction = self.satisfaction
            self.y_hat = tf.concat([self.satisfaction, 1 - self.satisfaction], -1)
            # self.probability = tf.nn.sigmoid(self.probability)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-9, 1.0)) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

class Model_DSPN_noAction(Model):
    def get_w(self, sess, inps):
        www = sess.run(self.www, feed_dict={
                self.y_ph: inps[0],
                self.adgroup_batch_ph: inps[2],
                self.member_batch_ph: inps[3],
                self.campaign_batch_ph: inps[4],
                self.item_batch_ph: inps[5],
                self.item_price_batch_ph: inps[6],
                self.cate_batch_ph: inps[7],
                self.commodity_batch_ph: inps[8],
                self.node_batch_ph: inps[9],
                self.effect_batch_his_id_ph: inps[10],
                self.effect_batch_his_val_ph: inps[11],
                self.effect_batch_his_mask_ph: inps[12],
                self.pos_batch_his_id_ph: inps[13],
                self.pos_batch_his_val_ph: inps[14],
                self.pos_batch_his_mask_ph: inps[15],
                self.direct_batch_his_id_ph: inps[16],
                self.direct_batch_his_val_ph: inps[17],
                self.direct_batch_his_mask_ph: inps[18],
                self.direct_action_batch_his_id_ph: inps[19],
                self.direct_action_batch_his_val_ph: inps[20],
                self.direct_action_batch_his_mask_ph: inps[21],
                self.pos_action_batch_his_id_ph: inps[22],
                self.pos_action_batch_his_val_ph: inps[23],
                self.pos_action_batch_his_mask_ph: inps[24]
            })
        return www

    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)
        
        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2
            self.ad_eb = tf.tile(self.ad_eb, [1, tf.shape(self.effect_batch_his_embedded)[1]])
            self.ad_eb = tf.reshape(self.ad_eb, [tf.shape(self.ad_eb)[0], -1, EMBEDDING_DIM])

            # direct info: sum pooling
            tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
            self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

            self.state_action_info = tf.concat([self.ad_eb, self.effect_batch_his_embedded], -1)

            self.state_action_info_bn = tf.layers.batch_normalization(inputs=self.state_action_info, name='state_action_info_bn')
            
            bi_rnn_outputs1, bi_rnn_states1 = tf.nn.bidirectional_dynamic_rnn(GRUCell(EMBEDDING_DIM), GRUCell(EMBEDDING_DIM), inputs=self.state_action_info_bn, \
                                       dtype=tf.float32, scope='gru1')
            bi_rnn_inputs2 = tf.concat([bi_rnn_outputs1[0], bi_rnn_outputs1[1]], -1)
            rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(GRUCell(n_effect + 1), GRUCell(n_effect + 1), inputs=bi_rnn_inputs2, \
                                       dtype=tf.float32, scope='gru2')
            
            self.intent = rnn_states2[0][:, 0 : n_effect] + rnn_states2[1][ :, 0 : n_effect]
            self.bias = rnn_states2[0][ :, -1] + rnn_states2[1][ :, -1]
            self.bias = tf.reshape(self.bias, [-1, 1])
            self.www = rnn_states2[0] + rnn_states2[1]
        
        with tf.name_scope('satisfaction'):
            self.intent = tf.tile(self.intent, [1, tf.shape(self.effect_batch_his_val_ph)[1]]) #
            self.intent = tf.reshape(self.intent, [tf.shape(self.effect_batch_his_val_ph)[0], tf.shape(self.effect_batch_his_val_ph)[1], tf.shape(self.effect_batch_his_val_ph)[2]]) 
            # self.effect_batch_his_val_ph: batch * day * effect
            self.satisfaction = tf.matmul(tf.expand_dims(self.effect_batch_his_val_ph, -2), tf.expand_dims(self.intent, -1))
            self.satisfaction = tf.reshape(self.satisfaction, [tf.shape(self.satisfaction)[0], tf.shape(self.satisfaction)[1]])
            self.satisfaction = self.satisfaction + self.bias
            self.satisfaction = tf.sigmoid(self.satisfaction) / tf.to_float(tf.shape(self.satisfaction)[1])
            self.satisfaction = tf.reduce_sum(self.satisfaction, -1)
            self.satisfaction = tf.reshape(self.satisfaction, [-1, 1])
            self.satisfaction = self.satisfaction
            self.y_hat = tf.concat([self.satisfaction, 1 - self.satisfaction], -1)
            # self.probability = tf.nn.sigmoid(self.probability)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-9, 1.0)) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

class Model_DSPN_noReport(Model):
    def DAVN_att_self_attention(self, facts, mask):

        # facts: batch * day * maxlen * embedding
        # mask: batch * day * maxlen
        tmp_mask = tf.expand_dims(mask, -1)
        tmp_mask = tf.matmul(tmp_mask, tf.transpose(tmp_mask, [0, 1, 3, 2]))
        key_mask = tf.equal(tmp_mask, tf.ones_like(tmp_mask))
        tmp_mat = tf.matmul(facts, tf.transpose(facts, [0, 1, 3, 2]))
        paddings = tf.ones_like(tmp_mat) * (-2 ** 32 + 1)
        tmp_mat = tf.where(key_mask, tmp_mat, paddings)
        tmp_mat = tf.nn.softmax(tmp_mat, name='alphas')
        output = tf.matmul(tmp_mat, facts)
        output = output * tf.expand_dims(mask, -1)
        return output

    def DAVN_att_ad_attention(self, query, facts, mask, EMBEDDING_DIM, type='null'):
        
        # facts: batch * day * maxlen * embedding
        # query: batch * embedding
        query = tf.tile(query, [1, tf.shape(facts)[1]])
        query = tf.reshape(query, [tf.shape(facts)[0], tf.shape(facts)[1], EMBEDDING_DIM])

        # direct info: sum pooling
        tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
        self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)
        query = tf.concat([self.davn_direct_batch_his_embedded, self.pos_batch_his_embedded, query], -1)
        with tf.name_scope('DAVN_att_ad_attention_ad_feature_embedding'):
            ad_att_dnn1 = tf.layers.dense(query, EMBEDDING_DIM * 2, activation=None)
            ad_att_dnn1 = prelu(ad_att_dnn1, 'ad_att_dnn1_' + type)
            ad_att_dnn2 = tf.layers.dense(ad_att_dnn1, EMBEDDING_DIM, activation=None)
            ad_att_dnn2 = prelu(ad_att_dnn2, 'ad_att_dnn2_' + type)
        query = ad_att_dnn2

        tmp_weight = tf.matmul(facts, tf.expand_dims(query, -1)) # batch * day * maxlen * 1
        tmp_weight = tf.reshape(tmp_weight, [tf.shape(tmp_weight)[0], \
                        tf.shape(tmp_weight)[1], tf.shape(tmp_weight)[2]])
        key_mask = tf.equal(mask, tf.ones_like(mask))
        paddings = tf.ones_like(tmp_weight) * (-2 * 32 + 1)
        tmp_weight = tf.where(key_mask, tmp_weight, paddings)
        tmp_weight = tf.nn.softmax(tmp_weight, name='alphas')
        output = facts * tf.expand_dims(tmp_weight, -1)
        output = tf.reduce_sum(output, 2)
        return output


    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)
        
        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2

        with tf.name_scope('direct_actions'):
            self.direct_action_his_info = self.DAVN_att_self_attention(self.direct_action_batch_his_embedded, self.direct_action_batch_his_mask_ph)
            self.direct_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.direct_action_his_info, self.direct_action_batch_his_mask_ph, EMBEDDING_DIM, type='direct')

        with tf.name_scope('pos_actions'):
            self.pos_action_his_info = self.DAVN_att_self_attention(self.pos_action_batch_his_embedded, self.pos_action_batch_his_mask_ph)
            self.pos_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.pos_action_his_info, self.pos_action_batch_his_mask_ph, EMBEDDING_DIM, type='pos')

        with tf.name_scope('rnn'):

            self.action_info = tf.concat([self.direct_action_info, self.pos_action_info], -1)
            self.action_info = tf.reshape(self.action_info, [tf.shape(self.action_info)[0], tf.shape(self.action_info)[1], EMBEDDING_DIM * 2])

            # direct info: sum pooling
            tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
            self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

            self.state_action_info = tf.concat([self.pos_batch_his_embedded, self.davn_direct_batch_his_embedded, self.action_info], -1)

            self.state_action_info_bn = tf.layers.batch_normalization(inputs=self.state_action_info, name='state_action_info_bn')
            
            bi_rnn_outputs1, bi_rnn_states1 = tf.nn.bidirectional_dynamic_rnn(GRUCell(EMBEDDING_DIM), GRUCell(EMBEDDING_DIM), inputs=self.state_action_info_bn, \
                                       dtype=tf.float32, scope='gru1')
            bi_rnn_inputs2 = tf.concat([bi_rnn_outputs1[0], bi_rnn_outputs1[1]], -1)
            rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(GRUCell(n_effect + 1), GRUCell(n_effect + 1), inputs=bi_rnn_inputs2, \
                                       dtype=tf.float32, scope='gru2')
            
            self.intent = rnn_states2[0][:, 0 : n_effect] + rnn_states2[1][ :, 0 : n_effect]
            self.bias = rnn_states2[0][ :, -1] + rnn_states2[1][ :, -1]
            self.bias = tf.reshape(self.bias, [-1, 1])
        
        with tf.name_scope('satisfaction'):
            self.intent = tf.tile(self.intent, [1, tf.shape(self.effect_batch_his_val_ph)[1]]) #
            self.intent = tf.reshape(self.intent, [tf.shape(self.effect_batch_his_val_ph)[0], tf.shape(self.effect_batch_his_val_ph)[1], tf.shape(self.effect_batch_his_val_ph)[2]]) 
            # self.effect_batch_his_val_ph: batch * day * effect
            self.satisfaction = tf.matmul(tf.expand_dims(self.effect_batch_his_val_ph, -2), tf.expand_dims(self.intent, -1))
            self.satisfaction = tf.reshape(self.satisfaction, [tf.shape(self.satisfaction)[0], tf.shape(self.satisfaction)[1]])
            self.satisfaction = self.satisfaction + self.bias
            self.satisfaction = tf.sigmoid(self.satisfaction) / tf.to_float(tf.shape(self.satisfaction)[1])
            self.satisfaction = tf.reduce_sum(self.satisfaction, -1)
            self.satisfaction = tf.reshape(self.satisfaction, [-1, 1])
            self.satisfaction = self.satisfaction
            self.y_hat = tf.concat([self.satisfaction, 1 - self.satisfaction], -1)
            # self.probability = tf.nn.sigmoid(self.probability)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-9, 1.0)) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

class Model_DSPN_no_intent(Model):
    def DAVN_att_self_attention(self, facts, mask):

        # facts: batch * day * maxlen * embedding
        # mask: batch * day * maxlen
        tmp_mask = tf.expand_dims(mask, -1)
        tmp_mask = tf.matmul(tmp_mask, tf.transpose(tmp_mask, [0, 1, 3, 2]))
        key_mask = tf.equal(tmp_mask, tf.ones_like(tmp_mask))
        tmp_mat = tf.matmul(facts, tf.transpose(facts, [0, 1, 3, 2]))
        paddings = tf.ones_like(tmp_mat) * (-2 ** 32 + 1)
        tmp_mat = tf.where(key_mask, tmp_mat, paddings)
        tmp_mat = tf.nn.softmax(tmp_mat, name='alphas')
        output = tf.matmul(tmp_mat, facts)
        output = output * tf.expand_dims(mask, -1)
        return output

    def DAVN_att_ad_attention(self, query, facts, mask, EMBEDDING_DIM, type='null'):
        
        # facts: batch * day * maxlen * embedding
        # query: batch * embedding
        query = tf.tile(query, [1, tf.shape(facts)[1]])
        query = tf.reshape(query, [tf.shape(facts)[0], tf.shape(facts)[1], EMBEDDING_DIM])

        # direct info: sum pooling
        tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
        self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)
        query = tf.concat([self.effect_batch_his_embedded, self.davn_direct_batch_his_embedded, self.pos_batch_his_embedded, query], -1)
        with tf.name_scope('DAVN_att_ad_attention_ad_feature_embedding'):
            ad_att_dnn1 = tf.layers.dense(query, EMBEDDING_DIM * 2, activation=None)
            ad_att_dnn1 = prelu(ad_att_dnn1, 'ad_att_dnn1_' + type)
            ad_att_dnn2 = tf.layers.dense(ad_att_dnn1, EMBEDDING_DIM, activation=None)
            ad_att_dnn2 = prelu(ad_att_dnn2, 'ad_att_dnn2_' + type)
        query = ad_att_dnn2

        tmp_weight = tf.matmul(facts, tf.expand_dims(query, -1)) # batch * day * maxlen * 1
        tmp_weight = tf.reshape(tmp_weight, [tf.shape(tmp_weight)[0], \
                        tf.shape(tmp_weight)[1], tf.shape(tmp_weight)[2]])
        key_mask = tf.equal(mask, tf.ones_like(mask))
        paddings = tf.ones_like(tmp_weight) * (-2 * 32 + 1)
        tmp_weight = tf.where(key_mask, tmp_weight, paddings)
        tmp_weight = tf.nn.softmax(tmp_weight, name='alphas')
        output = facts * tf.expand_dims(tmp_weight, -1)
        output = tf.reduce_sum(output, 2)
        return output


    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)
        
        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2

        with tf.name_scope('direct_actions'):
            self.direct_action_his_info = self.DAVN_att_self_attention(self.direct_action_batch_his_embedded, self.direct_action_batch_his_mask_ph)
            self.direct_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.direct_action_his_info, self.direct_action_batch_his_mask_ph, EMBEDDING_DIM, type='direct')

        with tf.name_scope('pos_actions'):
            self.pos_action_his_info = self.DAVN_att_self_attention(self.pos_action_batch_his_embedded, self.pos_action_batch_his_mask_ph)
            self.pos_action_info = self.DAVN_att_ad_attention(self.ad_eb, self.pos_action_his_info, self.pos_action_batch_his_mask_ph, EMBEDDING_DIM, type='pos')

        with tf.name_scope('rnn'):

            self.action_info = tf.concat([self.direct_action_info, self.pos_action_info], -1)
            self.action_info = tf.reshape(self.action_info, [tf.shape(self.action_info)[0], tf.shape(self.action_info)[1], EMBEDDING_DIM * 2])

            # direct info: sum pooling
            tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
            self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

            self.state_action_info = tf.concat([self.effect_batch_his_embedded, self.pos_batch_his_embedded, self.davn_direct_batch_his_embedded, self.action_info], -1)

            self.state_action_info_bn = tf.layers.batch_normalization(inputs=self.state_action_info, name='state_action_info_bn')
            
            bi_rnn_outputs1, bi_rnn_states1 = tf.nn.bidirectional_dynamic_rnn(GRUCell(EMBEDDING_DIM), GRUCell(EMBEDDING_DIM), inputs=self.state_action_info_bn, \
                                       dtype=tf.float32, scope='gru1')
            bi_rnn_inputs2 = tf.concat([bi_rnn_outputs1[0], bi_rnn_outputs1[1]], -1)
            rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(GRUCell(n_effect + 1), GRUCell(n_effect + 1), inputs=bi_rnn_inputs2, \
                                       dtype=tf.float32, scope='gru2')
            
            ad_dnn1 = tf.layers.dense(rnn_states2[0] + rnn_states2[1], 10, activation=None, name='davn_dnn1')
            ad_dnn1 = prelu(ad_dnn1, 'davn_prelu1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, 2, activation=None, name='davn_dnn2')
            self.y_hat = tf.nn.softmax(ad_dnn2, name='davn_softmax')

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-9, 1.0)) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

class Model_DSPN_ID(Model):
    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)
        
        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2
            self.ad_eb = tf.tile(self.ad_eb, [1, tf.shape(self.effect_batch_his_embedded)[1]])
            self.ad_eb = tf.reshape(self.ad_eb, [tf.shape(self.ad_eb)[0], -1, EMBEDDING_DIM])

        with tf.name_scope('rnn'):
            self.state_action_info = self.ad_eb

            self.state_action_info_bn = tf.layers.batch_normalization(inputs=self.state_action_info, name='state_action_info_bn')
            
            bi_rnn_outputs1, bi_rnn_states1 = tf.nn.bidirectional_dynamic_rnn(GRUCell(EMBEDDING_DIM), GRUCell(EMBEDDING_DIM), inputs=self.state_action_info_bn, \
                                       dtype=tf.float32, scope='gru1')
            bi_rnn_inputs2 = tf.concat([bi_rnn_outputs1[0], bi_rnn_outputs1[1]], -1)
            rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(GRUCell(n_effect + 1), GRUCell(n_effect + 1), inputs=bi_rnn_inputs2, \
                                       dtype=tf.float32, scope='gru2')
            
            self.intent = rnn_states2[0][:, 0 : n_effect] + rnn_states2[1][ :, 0 : n_effect]
            self.bias = rnn_states2[0][ :, -1] + rnn_states2[1][ :, -1]
            self.bias = tf.reshape(self.bias, [-1, 1])

            self.www = self.intent
        
        with tf.name_scope('satisfaction'):
            self.intent = tf.tile(self.intent, [1, tf.shape(self.effect_batch_his_val_ph)[1]]) #
            self.intent = tf.reshape(self.intent, [tf.shape(self.effect_batch_his_val_ph)[0], tf.shape(self.effect_batch_his_val_ph)[1], tf.shape(self.effect_batch_his_val_ph)[2]]) 
            # self.effect_batch_his_val_ph: batch * day * effect
            self.satisfaction = tf.matmul(tf.expand_dims(self.effect_batch_his_val_ph, -2), tf.expand_dims(self.intent, -1))
            self.satisfaction = tf.reshape(self.satisfaction, [tf.shape(self.satisfaction)[0], tf.shape(self.satisfaction)[1]])
            self.satisfaction = self.satisfaction + self.bias
            self.satisfaction = tf.sigmoid(self.satisfaction) / tf.to_float(tf.shape(self.satisfaction)[1])
            self.satisfaction = tf.reduce_sum(self.satisfaction, -1)
            self.satisfaction = tf.reshape(self.satisfaction, [-1, 1])
            self.satisfaction = self.satisfaction
            self.y_hat = tf.concat([self.satisfaction, 1 - self.satisfaction], -1)
            # self.probability = tf.nn.sigmoid(self.probability)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-9, 1.0)) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

class Model_DSPN_Report(Model):
    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)
        
        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2

        with tf.name_scope('rnn'):

            # self.state_action_info = tf.concat([self.effect_batch_his_embedded, self.pos_batch_his_embedded, self.davn_direct_batch_his_embedded, self.action_info], -1)
            self.state_action_info = self.effect_batch_his_embedded

            self.state_action_info_bn = tf.layers.batch_normalization(inputs=self.state_action_info, name='state_action_info_bn')
            
            bi_rnn_outputs1, bi_rnn_states1 = tf.nn.bidirectional_dynamic_rnn(GRUCell(EMBEDDING_DIM), GRUCell(EMBEDDING_DIM), inputs=self.state_action_info_bn, \
                                       dtype=tf.float32, scope='gru1')
            bi_rnn_inputs2 = tf.concat([bi_rnn_outputs1[0], bi_rnn_outputs1[1]], -1)
            rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(GRUCell(n_effect + 1), GRUCell(n_effect + 1), inputs=bi_rnn_inputs2, \
                                       dtype=tf.float32, scope='gru2')
            
            self.intent = rnn_states2[0][:, 0 : n_effect] + rnn_states2[1][ :, 0 : n_effect]
            self.bias = rnn_states2[0][ :, -1] + rnn_states2[1][ :, -1]
            self.bias = tf.reshape(self.bias, [-1, 1])

            self.www = self.intent
        
        with tf.name_scope('satisfaction'):
            self.intent = tf.tile(self.intent, [1, tf.shape(self.effect_batch_his_val_ph)[1]]) #
            self.intent = tf.reshape(self.intent, [tf.shape(self.effect_batch_his_val_ph)[0], tf.shape(self.effect_batch_his_val_ph)[1], tf.shape(self.effect_batch_his_val_ph)[2]]) 
            # self.effect_batch_his_val_ph: batch * day * effect
            self.satisfaction = tf.matmul(tf.expand_dims(self.effect_batch_his_val_ph, -2), tf.expand_dims(self.intent, -1))
            self.satisfaction = tf.reshape(self.satisfaction, [tf.shape(self.satisfaction)[0], tf.shape(self.satisfaction)[1]])
            self.satisfaction = self.satisfaction + self.bias
            self.satisfaction = tf.sigmoid(self.satisfaction) / tf.to_float(tf.shape(self.satisfaction)[1])
            self.satisfaction = tf.reduce_sum(self.satisfaction, -1)
            self.satisfaction = tf.reshape(self.satisfaction, [-1, 1])
            self.satisfaction = self.satisfaction
            self.y_hat = tf.concat([self.satisfaction, 1 - self.satisfaction], -1)
            # self.probability = tf.nn.sigmoid(self.probability)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-9, 1.0)) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

class Model_DSPN_Action(Model):
    def __init__(self, n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE):
        super().__init__(n_adgroup, n_member, n_campaign, \
                       n_item, n_cate, n_commodity, \
                       n_node, n_effect, n_pos, n_direct, \
                       n_direct_action, n_pos_action, \
                       EMBEDDING_DIM, ATTENTION_SIZE)
        
        with tf.name_scope('Ad_feature_embedding'):

            ad_feature = tf.concat([ \
                    # self.adgroup_batch_embedded, \
                    self.member_batch_embedded, \
                    # self.campaign_batch_embedded, \
                    # self.item_batch_embedded, \
                    self.cate_batch_embedded, \
                    self.commodity_batch_embedded, \
                    self.node_batch_embedded \
                    ], -1)

            ad_dnn1 = tf.layers.dense(ad_feature, EMBEDDING_DIM * 2, activation=None, name='ad_feature_f1')
            ad_dnn1 = prelu(ad_dnn1, 'ad_feature_p1')
            ad_dnn2 = tf.layers.dense(ad_dnn1, EMBEDDING_DIM, activation=None, name='ad_feature_f2')
            ad_dnn2 = prelu(ad_dnn2, 'ad_feature_p2')
            self.ad_eb = ad_dnn2

        with tf.name_scope('rnn'):
            # direct action info: sum pooling
            tmp_direct_action_batch_his_mask_ph = tf.expand_dims(self.direct_action_batch_his_mask_ph, -1)
            self.dspn_action_direct_action_batch_his_embedded = tf.reduce_sum(tmp_direct_action_batch_his_mask_ph * self.direct_action_batch_his_embedded, 2)

            # pos action info: sum pooling
            tmp_pos_action_batch_his_mask_ph = tf.expand_dims(self.pos_action_batch_his_mask_ph, -1)
            self.dspn_action_pos_action_batch_his_embedded = tf.reduce_sum(tmp_pos_action_batch_his_mask_ph * self.pos_action_batch_his_embedded, 2)

            # direct info: sum pooling
            tmp_direct_batch_his_mask_ph = tf.expand_dims(self.direct_batch_his_mask_ph, -1)
            self.davn_direct_batch_his_embedded = tf.reduce_sum(tmp_direct_batch_his_mask_ph * self.direct_batch_his_embedded, 2)

            self.state_action_info = tf.concat([self.pos_batch_his_embedded, self.davn_direct_batch_his_embedded, self.dspn_action_pos_action_batch_his_embedded, self.dspn_action_direct_action_batch_his_embedded], -1)
            # self.state_action_info = self.effect_batch_his_embedded

            self.state_action_info_bn = tf.layers.batch_normalization(inputs=self.state_action_info, name='state_action_info_bn')
            
            bi_rnn_outputs1, bi_rnn_states1 = tf.nn.bidirectional_dynamic_rnn(GRUCell(EMBEDDING_DIM), GRUCell(EMBEDDING_DIM), inputs=self.state_action_info_bn, \
                                       dtype=tf.float32, scope='gru1')
            bi_rnn_inputs2 = tf.concat([bi_rnn_outputs1[0], bi_rnn_outputs1[1]], -1)
            rnn_outputs2, rnn_states2 = tf.nn.bidirectional_dynamic_rnn(GRUCell(n_effect + 1), GRUCell(n_effect + 1), inputs=bi_rnn_inputs2, \
                                       dtype=tf.float32, scope='gru2')
            
            self.intent = rnn_states2[0][:, 0 : n_effect] + rnn_states2[1][ :, 0 : n_effect]
            self.bias = rnn_states2[0][ :, -1] + rnn_states2[1][ :, -1]
            self.bias = tf.reshape(self.bias, [-1, 1])

            self.www = self.intent
        
        with tf.name_scope('satisfaction'):
            self.intent = tf.tile(self.intent, [1, tf.shape(self.effect_batch_his_val_ph)[1]]) #
            self.intent = tf.reshape(self.intent, [tf.shape(self.effect_batch_his_val_ph)[0], tf.shape(self.effect_batch_his_val_ph)[1], tf.shape(self.effect_batch_his_val_ph)[2]]) 
            # self.effect_batch_his_val_ph: batch * day * effect
            self.satisfaction = tf.matmul(tf.expand_dims(self.effect_batch_his_val_ph, -2), tf.expand_dims(self.intent, -1))
            self.satisfaction = tf.reshape(self.satisfaction, [tf.shape(self.satisfaction)[0], tf.shape(self.satisfaction)[1]])
            self.satisfaction = self.satisfaction + self.bias
            self.satisfaction = tf.sigmoid(self.satisfaction) / tf.to_float(tf.shape(self.satisfaction)[1])
            self.satisfaction = tf.reduce_sum(self.satisfaction, -1)
            self.satisfaction = tf.reshape(self.satisfaction, [-1, 1])
            self.satisfaction = self.satisfaction
            self.y_hat = tf.concat([self.satisfaction, 1 - self.satisfaction], -1)
            # self.probability = tf.nn.sigmoid(self.probability)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = self.y_ph
            loss = -tf.reduce_mean(tf.log(tf.clip_by_value(self.y_hat, 1e-9, 1.0)) * self.target_ph)
            self.loss = loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()