import tensorflow as tf
import numpy as np
import sys
import random
import datetime
from get_data import *
from prepare_data import *
from model import *
from utils import *
from matplotlib import pyplot as plt
plt.switch_backend('Agg')

import logging

logging.basicConfig(
    level=logging.INFO,
    filename='model.log',
    format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s"
)


EMBEDDING_DIM = 18
ATTENTION_SIZE = 18 * 2
best_auc = 0.0
best_acc = 0.0

test_iterations = []
test_loss_ = []
test_accuracy_ = []
test_auc_ = []

def get_w(sess, test_data, model, model_path):
    tot_w = ''
    for raw_data in test_data:
        y, y_, adgroup, member, campaign, item, item_price, cate, commodity, node, \
        effect_id, effect, effect_mask, \
        pos_id, pos, pos_mask, \
        direct_id, direct, direct_mask, \
        direct_action_type, direct_action_value, direct_action_mask, \
        pos_action_type, pos_action_value, pos_action_mask = prepare_data(raw_data)
        w = model.get_w(sess, \
                        [y, y_, adgroup, member, campaign, item, item_price, cate, commodity, node, \
                        effect_id, effect, effect_mask, \
                        pos_id, pos, pos_mask, \
                        direct_id, direct, direct_mask, \
                        direct_action_type, direct_action_value, direct_action_mask, \
                        pos_action_type, pos_action_value, pos_action_mask])
        w = w.tolist()
        for x, label, z in zip(w, y, effect):
            w_w = '['
            for v in x:
                w_w = w_w + str(v) + ','
            w_w = w_w[0: -1]
            w_w = w_w + ']'
            w_w = w_w + '+' + str(label[0]) 
            w_w = w_w + '+' + '['
            for v in z:
                w_w = w_w + '(' 
                for vv in v:
                    vv = vv.tolist()
                    w_w = w_w + str(vv) + ','
                w_w = w_w[0: -1]
                w_w = w_w + ')' + '$'
            w_w = w_w[0: -1]
            w_w = w_w + ']' + '\n'
            tot_w = tot_w + w_w
    print(tot_w)
    logging.info(tot_w)

def eval(sess, test_data, model, model_path):
    loss_sum = 0.0
    accuracy_sum = 0.0
    aux_loss_sum = 0.0
    nums = 0
    stored_arr = []
    for raw_data in test_data:
        nums += 1
        y, y_, adgroup, member, campaign, item, item_price, cate, commodity, node, \
        effect_id, effect, effect_mask, \
        pos_id, pos, pos_mask, \
        direct_id, direct, direct_mask, \
        direct_action_type, direct_action_value, direct_action_mask, \
        pos_action_type, pos_action_value, pos_action_mask = prepare_data(raw_data)
        prob, loss, acc = \
            model.calculate(sess, \
                           [y, y_, adgroup, member, campaign, item, item_price, cate, commodity, node, \
                            effect_id, effect, effect_mask, \
                            pos_id, pos, pos_mask, \
                            direct_id, direct, direct_mask, \
                            direct_action_type, direct_action_value, direct_action_mask, \
                            pos_action_type, pos_action_value, pos_action_mask])
        loss_sum += loss
        accuracy_sum += acc
        
        prob_1 = prob[:, 0].tolist()
        target = y
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])

    test_auc = calc_auc(stored_arr)
    accuracy_mean = accuracy_sum / nums
    loss_mean = loss_sum / nums
    global best_auc
    global best_acc
    if best_acc < accuracy_mean:
        best_acc = accuracy_mean
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
    return test_auc, loss_mean, accuracy_mean

def draw(path):
    # test auc
    plt.figure()
    plt.plot(test_iterations, test_auc_)
    plt.title('test auc')
    plt.savefig(path+'_test_auc')

    # test loss
    plt.figure()
    plt.plot(test_iterations, test_loss_)
    plt.title('test loss')
    plt.savefig(path+'_test_loss')

    # test accuracy
    plt.figure()
    plt.plot(test_iterations, test_accuracy_)
    plt.title('test accuracy')
    plt.savefig(path+'_test_accuracy')


def train(
    file='ad_action_state',
    batch_size=32,
    test_iter=100,
    save_iter=1000000000,
    model_type='DSPN',
    seed=3,
    shuffle_each_epoch=True,
):
    is_shuffle = 'noshuffle_'
    if shuffle_each_epoch:
        is_shuffle = 'shuffle_'
    model_path = model_type + '_best_model/' + is_shuffle + '_' + str(seed)

    logging.getLogger('matplotlib.font_manager').disabled = True
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print("data preparation...")
        train_data, test_data = getData(file=file, 
                                        batch_size=batch_size, 
                                        shuffle_each_epoch=shuffle_each_epoch)
        print("data ready.")
        n_adgroup, n_member, n_campaign, \
        n_item, n_cate, n_commodity, \
        n_node, n_effect, n_pos, n_direct, \
        n_direct_action, n_pos_action = get_n()
        print(get_n())
        logging.info(get_n())
        if model_type == 'MLP':
            model = Model_MLP(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'WideDeep':
            model = Model_WideDeep(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'RNN':
            model = Model_RNN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'biRNN':
            model = Model_biRNN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN':
            model = Model_DSPN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_RNN':
            model = Model_DSPN_RNN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_MLP':
            model = Model_DSPN_MLP(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_no_att':
            model = Model_DSPN_no_att(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_noID':
            model = Model_DSPN_noID(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_noAction':
            model = Model_DSPN_noAction(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_noReport':
            model = Model_DSPN_noReport(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_no_intent':
            model = Model_DSPN_no_intent(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_ID':
            model = Model_DSPN_ID(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_Action':
            model = Model_DSPN_Action(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_Report':
            model = Model_DSPN_Report(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        else:
            print("Invalid model_type: %s" % (model_type))
            return
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        t_auc, t_loss, t_accuracy = eval(sess, test_data, model, model_path)

        logging.info('model: %s, test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f' % (model_type, t_auc, t_loss, t_accuracy))
        sys.stdout.flush()
        print('test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f' % (t_auc, t_loss, t_accuracy))
        sys.stdout.flush()

        iter = 0
        lr = 0.001
        loss_sum = 0.0
        accuracy_sum = 0.0
        eval_sum = 0.0
        path = 'result_plot_' + model_type + '_batch_size_' + str(batch_size) + '_iter_num_' + str(test_iter)
        global best_auc
        global best_acc
        for itr in range(3):
            print("start iteration %d." % (itr + 1))
            for raw_data in train_data:
                y, y_, adgroup, member, campaign, item, item_price, cate, commodity, node, \
                effect_id, effect, effect_mask, \
                pos_id, pos, pos_mask, \
                direct_id, direct, direct_mask, \
                direct_action_type, direct_action_value, direct_action_mask, \
                pos_action_type, pos_action_value, pos_action_mask = prepare_data(raw_data)
                loss, acc = \
                    model.train(sess, \
                               [lr, y, y_, adgroup, member, campaign, item, item_price, cate, commodity, node, \
                                effect_id, effect, effect_mask, \
                                pos_id, pos, pos_mask, \
                                direct_id, direct, direct_mask, \
                                direct_action_type, direct_action_value, direct_action_mask, \
                                pos_action_type, pos_action_value, pos_action_mask])
                loss_sum += loss
                accuracy_sum += acc
                iter += 1
                sys.stdout.flush()
                if iter % test_iter == 0:

                    t_auc, t_loss, t_accuracy = eval(sess, test_data, model, model_path)
                    
                    logging.info('model: %s, iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f' % \
                          (model_type, iter, loss_sum / test_iter, accuracy_sum / test_iter))
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f' % \
                          (iter, loss_sum / test_iter, accuracy_sum / test_iter))
                    logging.info('model: %s, test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f' % \
                          (model_type, t_auc, t_loss, t_accuracy))
                    print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f' % \
                          (t_auc, t_loss, t_accuracy))
                    logging.info("best_auc = %f" % best_auc)
                    print("best_auc = %f" % best_auc)
                    logging.info("best_acc = %f" % best_acc)
                    print("best_acc = %f" % best_acc)
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                    test_iterations.append(len(test_iterations))
                    test_auc_.append(t_auc)
                    test_loss_.append(t_loss)
                    test_accuracy_.append(t_accuracy)
                if iter % save_iter == 0:
                    logging.info('save model iter: %d' % iter)
                    print('save model iter: %d' % iter)
                    model.save(sess, model_path + "--" + str(iter))
            lr *= 0.5
        draw(path)
        logging.info(model_path)
        logging.info(path)
        logging.info(test_iterations)
        logging.info(test_auc_)
        logging.info(test_loss_)
        logging.info(test_accuracy_)
        logging.info("best_auc = %f" % best_auc)
        print("best_auc = %f" % best_auc)
        logging.info("best_acc = %f" % best_acc)
        print("best_acc = %f" % best_acc)

def test(
    file='ad_action_state',
    batch_size=32,
    model_type='DSPN',
    seed=3,
    shuffle_each_epoch=True
):
    is_shuffle = 'noshuffle_'
    if shuffle_each_epoch:
        is_shuffle = 'shuffle_'
    model_path = model_type + '_best_model/' + is_shuffle + '_' + str(seed)

    logging.getLogger('matplotlib.font_manager').disabled = True
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print("data preparation...")
        train_data, test_data = getData(file=file, 
                                        batch_size=batch_size, 
                                        shuffle_each_epoch=shuffle_each_epoch)
        print("data ready.")
        n_adgroup, n_member, n_campaign, \
        n_item, n_cate, n_commodity, \
        n_node, n_effect, n_pos, n_direct, \
        n_direct_action, n_pos_action = get_n()
        print(get_n())
        logging.info(get_n())
        if model_type == 'MLP':
            model = Model_MLP(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'WideDeep':
            model = Model_WideDeep(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'RNN':
            model = Model_RNN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'biRNN':
            model = Model_biRNN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN':
            model = Model_DSPN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_RNN':
            model = Model_DSPN_RNN(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_MLP':
            model = Model_DSPN_MLP(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_no_att':
            model = Model_DSPN_no_att(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_noID':
            model = Model_DSPN_noID(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_noAction':
            model = Model_DSPN_noAction(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_noReport':
            model = Model_DSPN_noReport(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_no_intent':
            model = Model_DSPN_no_intent(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_ID':
            model = Model_DSPN_ID(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_Action':
            model = Model_DSPN_Action(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        elif model_type == 'DSPN_Report':
            model = Model_DSPN_Report(n_adgroup, n_member, n_campaign, \
                              n_item, n_cate, n_commodity, \
                              n_node, n_effect, n_pos, n_direct, \
                              n_direct_action, n_pos_action, \
                              EMBEDDING_DIM, ATTENTION_SIZE)
        else:
            print("Invalid model_type: %s" % (model_type))
            return
        model.restore(sess, model_path)
        if model_type == 'DSPN' or model_type == 'DSPN_noAction':
            get_w(sess, test_data, model, model_path)
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f' % eval(sess, test_data, model, model_path))
        logging.info('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f' % eval(sess, test_data, model, model_path))

if __name__ == '__main__':
    if len(sys.argv) >= 4:
        SEED = int(sys.argv[3])
    else:
        SEED = 3
    logging.info("SEED = %d" % SEED)
    print("SEED = %d" % SEED)
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if sys.argv[1] == 'train':
        train(model_type=sys.argv[2], seed=SEED)
    elif sys.argv[1] == 'test':
        test(model_type=sys.argv[2], seed=SEED)
    else:
        logging.info('do nothing...')
        print('do nothing...')
