#coding=utf-8
# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import time



from tensorflow.python import pywrap_tensorflow

from monodepth_model_LF7x7Star import *
from dataloader_lf7x7star_zec import *
from average_gradients import *
from scipy import misc
from evalfunctions7x7 import *

train_or_test = False
parser = argparse.ArgumentParser(description='4D LF Monodepth with TensorFlow implementation.')

filenames_fileTest = '4dlffilenames_val7x7star.txt' #test or val
gt_path = 'evaluation_toolkit/data/eval_gt'
data_path ="evaluation_toolkit/data"

"""结果保存位置"""
output_directory = 'result'

log_directory = './../allresult_1/imageloss_refocusedlossMulti10_consistence0.001_450epochs_dots_2020'     #save the modle into this dir
pfmdir = output_directory
if not os.path.exists(output_directory):
    os.mkdir(output_directory)


checkpoint_path = 'model/model-85000'

retrain_checkpoint_path = './imageloss_resnet50'+'/monoLFdepth/model-5000'
# retrain_checkpoint_path = ""

parser.add_argument('--model_name', type=str, help='model name', default='monoLFdepth')
parser.add_argument('--data_path', type=str, help='path to the data', required=False, default=data_path)

parser.add_argument('--input_height', type=int, help='input height', default=512)
parser.add_argument('--input_width', type=int, help='input width', default=512)
parser.add_argument('--batch_size', type=int, help='batch size', default=4)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=300)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--dp_consistency_sigmoid_scale', type=float, help='scale for sigmoid function in dp_consist computation', default=10.)

parser.add_argument('--alpha_image_loss', type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_consistency_loss_weight', type=float, help='left-right consistency weight', default=0.1)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.)#0.01
parser.add_argument('--centerSymmetry_loss_weight', type=float, help='left-center-right consistency weight', default=1.)

parser.add_argument('--use_deconv', help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus', type=int, help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory', type=str,
                    help='output directory for test disparities, if empty outputs to checkpoint folder',
                    default=output_directory)
parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries',
                    default=log_directory)


if train_or_test:
    if not os.path.exists(log_directory):
        os.mkdir(log_directory)
    parser.add_argument('--mode',                      type=str,   help='train or test', default='test')
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=False,default=filenames_file)
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default=retrain_checkpoint_path)
    parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true', default=False)
    parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true',default=True)
else:
    parser.add_argument('--mode',                      type=str,   help='train or test', default='test')
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=False,default=filenames_fileTest)
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default=checkpoint_path)
    parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
    parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')


args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    x = 1.0 - l_mask - r_mask
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def train(params):
    """Training loop."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch

        boundaries = [np.int32((2/5) * num_total_steps), np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4, args.learning_rate / 8]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        # Optimizer
        opt_step = tf.train.AdamOptimizer(learning_rate)

        # loading data
        dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.mode)

        images_list = dataloader.image_batch_list
        images_splits_list = [tf.split(single, args.num_gpus, 0) for single in images_list]

        tower_grads  = []
        tower_losses = []
        reuse_variables = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):

                    images_splits = [single[i] for single in images_splits_list]
                    model = MonodepthModel(params, args.mode, images_splits, reuse_variables=reuse_variables, model_index=i)

                    loss = model.total_loss
                    tower_losses.append(loss)

                    reuse_variables = True

                    grads = opt_step.compute_gradients(loss)

                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

        total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver(max_to_keep=20)

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
            train_saver.restore(sess, args.checkpoint_path)

            if args.retrain:
                sess.run(global_step.assign(0))

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            _, loss_value = sess.run([apply_gradient_op, total_loss])
            duration = time.time() - before_op_time
            if step and step % 100 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 5000 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

        print('done.')


def test(params):
    """Test function."""
    dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.mode)
    center_image = [dataloader.center_image_batch]
    model = MonodepthModel(params, args.mode, center_image ,
                           reuse_variables=None,model_index=None)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path
    train_saver.restore(sess, restore_path)

    reader = pywrap_tensorflow.NewCheckpointReader(restore_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # reader.get
    """
    graph = tf.reset_default_graph()
    # keylist = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # keylist =graph.get_collection()
    # chek1 = tf.train.load_checkpoint(restore_path)


    reader1 = pywrap_tensorflow.NewCheckpointReader(checkpoint_path1)
    var_to_shape_map1 = reader.get_variable_to_shape_map()

    tensor_list = tf.contrib.graph_editor.get_tensors(tf.get_default_graph())
    with tf.Session() as sess:
        graph2 = sess.graph
        graph1 = tf.get_default_graph()
        for key in var_to_shape_map:
            checkpoint_data= (reader.get_tensor(key)+reader1.get_tensor(key))/2.
            key += ':0'
            # aaa = sess.graph.get_tensor_by_name(key)
            # op = graph.get_operation_by_name(key)

            op = graph1.get_tensor_by_name(key)

            assign_op = tf.assign(graph1.get_tensor_by_name(key),checkpoint_data)


    """
    num_test_samples = count_text_lines(args.filenames_file)
    # TEST_IMAGES =['dots'+str(i) for i in range(num_test_samples)]
    print('now testing {} files'.format(num_test_samples))
    disparities = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    for step in range(num_test_samples):
        disp = sess.run(model.disp_center_est)
        disparities[step] = disp[0].squeeze()
        disparities_pp[step] = post_process_disparity(disp.squeeze())
        origin_pfm_dir = os.path.join(pfmdir,"originpfm")
        pp_pfm_dir = os.path.join(pfmdir,"pppfm")
        if not os.path.exists(origin_pfm_dir):
            os.mkdir(origin_pfm_dir)
        if not os.path.exists(pp_pfm_dir):
            os.mkdir(pp_pfm_dir)
        write_pfm(disp[0, :, :, 0], origin_pfm_dir + '/%s.pfm' % TEST_IMAGES[step])
        write_pfm(disparities_pp[step], pp_pfm_dir + '/%s.pfm' % TEST_IMAGES[step])
        misc.imsave(pp_pfm_dir + '/' + TEST_IMAGES[step] + '_submit.jpg', disparities_pp[step])


    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy',    disparities)
    np.save(output_directory + '/disparities_pp.npy', disparities_pp)
    # misc.imsave('./' + 'disparities.png',disparities)
    # misc.imsave('./' + 'disparities_pp.png', disparities_pp)
    # for i in range(disparities.shape[0]):
    #     print ("origin result")
    #     myresult = open(output_directory + '/result.txt','a+')
    #     misc.imsave(output_directory+'/' + str(i) + 'dis.jpg', disparities[i, :, :])
    #     get_scores_file(disparities[i, :, :],i,myresult)
    #     # get_scores(disparities[i, :, :],i)
    #     print("pp result")
    #     misc.imsave(output_directory+'/' + str(i) + 'dispp.jpg', disparities_pp[i, :, :])
    #     get_scores_file(disparities_pp[i, :, :], i,myresult)
        # get_scores(disparities_pp[i, :, :], i)
    avg_score = 0.
    avg_score_pp = 0.
    myresult = open(output_directory + '/result.txt', 'a+')
    print("-----------------load checkpoint {}---------------".format(args.checkpoint_path))
    myresult.write("-----------------load checkpoint {}".format(args.checkpoint_path+"\n"))

    for i in range(disparities.shape[0]):
        print("origin result")

        # if i <=7:
        #     misc.imsave(output_directory + '/' + VAL_IMAGES[i] + '_disp.jpg', disparities[i, :, :])
        #     save_singledisp(disparities[i, :, :],output_directory,VAL_IMAGES[i]+'plt')
        #
        # else:
        #     misc.imsave(output_directory + '/' + TEST_IMAGES[i] + '_disp.jpg', disparities[i, :, :])
        #     save_singledisp(disparities[i, :, :], output_directory, TEST_IMAGES[i]+'plt')
        #     continue
        if i <=7:
            misc.imsave(output_directory + '/' + VAL_IMAGES[i] + '_disp.jpg', disparities_pp[i, :, :])
            save_singledisp(disparities_pp[i, :, :],output_directory,VAL_IMAGES[i]+'plt')

        else:
            misc.imsave(output_directory + '/' + TEST_IMAGES[i] + '_disp.jpg', disparities_pp[i, :, :])
            save_singledisp(disparities_pp[i, :, :], output_directory, TEST_IMAGES[i]+'plt')
            continue
        # misc.imsave(output_directory + '/' + VAL_IMAGES_ALL[i] + '_dis.jpg', disparities[i, :, :])
        # get_scores_file(disparities[i, :, :],i,myresult)
        gt = os.path.join(gt_path, VAL_IMAGES[i] + "/valid_disp_map.npy")
        # gt = os.path.join(gt_path, VAL_IMAGES_ALL[i] + "/valid_disp_map.npy")
        if not os.path.exists(gt):
            print("error path " + gt)
        gt_img = np.load(gt)
        #save_singledisp(gt_img, output_directory, VAL_IMAGES[i] + '_gt')
        error_img, error_score = get_scores_file(disparities[i, :, :], i, myresult)

        # error_img, error_score = get_scores_file_all(disparities[i, :, :], i, myresult)
        # save_erroplt_all(gt_img, disparities[i, :, :], error_img, output_directory, i, False)
        save_erroplt(gt_img, disparities[i, :, :], error_img, output_directory, i, False)
        avg_score += error_score
        # get_scores(disparities[i, :, :],i)
        print("pp result")
        misc.imsave(output_directory + '/' + VAL_IMAGES[i] + 'dispp.jpg', disparities_pp[i, :, :])
        error_img, error_score = get_scores_file(disparities_pp[i, :, :], i, myresult)

        save_singledisp_error(error_img, output_directory, VAL_IMAGES[i] + 'errorplt')
        avg_score_pp +=error_score
        save_erroplt(gt_img, disparities_pp[i, :, :], error_img, output_directory, i, True)

        # get_scores(disparities_pp[i, :, :], i)
    print("-----------------avg score {:.5f}---------------".format(100.-(avg_score) / 8))
    myresult.write("-----------------avg score {}---------------".format(str(100.-(avg_score) / 8))+"\n")

    print("-----------------avg score {:.5f}---------------".format(100. - (avg_score_pp) / 8))
    myresult.write("-----------------avg score {}---------------".format(str(100. - (avg_score_pp) / 8)) + "\n")

    print('done.')
def test_additional(params):
    """Test function."""
    dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.mode)
    center_image = [dataloader.center_image_batch]
    model = MonodepthModel(params, args.mode, center_image ,
                           reuse_variables=None,model_index=None)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path
    train_saver.restore(sess, restore_path)

    reader = pywrap_tensorflow.NewCheckpointReader(restore_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # reader.get
    """
    graph = tf.reset_default_graph()
    # keylist = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # keylist =graph.get_collection()
    # chek1 = tf.train.load_checkpoint(restore_path)


    reader1 = pywrap_tensorflow.NewCheckpointReader(checkpoint_path1)
    var_to_shape_map1 = reader.get_variable_to_shape_map()

    tensor_list = tf.contrib.graph_editor.get_tensors(tf.get_default_graph())
    with tf.Session() as sess:
        graph2 = sess.graph
        graph1 = tf.get_default_graph()
        for key in var_to_shape_map:
            checkpoint_data= (reader.get_tensor(key)+reader1.get_tensor(key))/2.
            key += ':0'
            # aaa = sess.graph.get_tensor_by_name(key)
            # op = graph.get_operation_by_name(key)

            op = graph1.get_tensor_by_name(key)

            assign_op = tf.assign(graph1.get_tensor_by_name(key),checkpoint_data)


    """
    num_test_samples = count_text_lines(args.filenames_file)
    # TEST_IMAGES =['dots'+str(i) for i in range(num_test_samples)]
    print('now testing {} files'.format(num_test_samples))
    disparities = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    for step in range(num_test_samples):
        disp = sess.run(model.disp_center_est)
        disparities[step] = disp[0].squeeze()
        a = disp.squeeze()
        aaa = post_process_disparity(a)
        disparities_pp[step] = post_process_disparity(disp.squeeze())
        origin_pfm_dir = os.path.join(pfmdir,"originpfm")
        pp_pfm_dir = os.path.join(pfmdir,"pppfm")
        if not os.path.exists(origin_pfm_dir):
            os.mkdir(origin_pfm_dir)
        if not os.path.exists(pp_pfm_dir):
            os.mkdir(pp_pfm_dir)
        write_pfm(disp[0, :, :, 0], origin_pfm_dir + '/%s.pfm' % ADDITIONAL_IMAGES_ALL[step])
        write_pfm(disparities_pp[step], pp_pfm_dir + '/%s.pfm' % ADDITIONAL_IMAGES_ALL[step])
        misc.imsave(pp_pfm_dir + '/' + ADDITIONAL_IMAGES_ALL[step] + '_submit.jpg', disparities_pp[step])
        misc.imsave(origin_pfm_dir + '/' + ADDITIONAL_IMAGES_ALL[step] + '_flip_left_right.jpg', a[1,:,:])


    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy',    disparities)
    np.save(output_directory + '/disparities_pp.npy', disparities_pp)
    # misc.imsave('./' + 'disparities.png',disparities)
    # misc.imsave('./' + 'disparities_pp.png', disparities_pp)
    # for i in range(disparities.shape[0]):
    #     print ("origin result")
    #     myresult = open(output_directory + '/result.txt','a+')
    #     misc.imsave(output_directory+'/' + str(i) + 'dis.jpg', disparities[i, :, :])
    #     get_scores_file(disparities[i, :, :],i,myresult)
    #     # get_scores(disparities[i, :, :],i)
    #     print("pp result")
    #     misc.imsave(output_directory+'/' + str(i) + 'dispp.jpg', disparities_pp[i, :, :])
    #     get_scores_file(disparities_pp[i, :, :], i,myresult)
        # get_scores(disparities_pp[i, :, :], i)
    avg_score = 0.
    avg_score_pp = 0.
    myresult = open(output_directory + '/result.txt', 'a+')
    print("-----------------load checkpoint {}---------------".format(args.checkpoint_path))
    myresult.write("-----------------load checkpoint {}".format(args.checkpoint_path+"\n"))

    for i in range(disparities.shape[0]):
        print("origin result")

        # if i <=7:
        #     misc.imsave(output_directory + '/' + VAL_IMAGES[i] + '_disp.jpg', disparities[i, :, :])
        #     save_singledisp(disparities[i, :, :],output_directory,VAL_IMAGES[i]+'plt')
        #
        # else:
        #     misc.imsave(output_directory + '/' + TEST_IMAGES[i] + '_disp.jpg', disparities[i, :, :])
        #     save_singledisp(disparities[i, :, :], output_directory, TEST_IMAGES[i]+'plt')
        #     continue
        if i <=7:
            misc.imsave(output_directory + '/' + ADDITIONAL_IMAGES_ALL[i] + '_disp.jpg', disparities_pp[i, :, :])
            save_singledisp(disparities_pp[i, :, :],output_directory,ADDITIONAL_IMAGES_ALL[i]+'plt')

        else:
            misc.imsave(output_directory + '/' + ADDITIONAL_IMAGES_ALL[i] + '_disp.jpg', disparities_pp[i, :, :])
            save_singledisp(disparities_pp[i, :, :], output_directory, ADDITIONAL_IMAGES_ALL[i]+'plt')
            continue
        # misc.imsave(output_directory + '/' + VAL_IMAGES_ALL[i] + '_dis.jpg', disparities[i, :, :])
        # get_scores_file(disparities[i, :, :],i,myresult)
        gt = os.path.join(gt_path, ADDITIONAL_IMAGES_ALL[i] + "/valid_disp_map.npy")
        # gt = os.path.join(gt_path, VAL_IMAGES_ALL[i] + "/valid_disp_map.npy")
        if not os.path.exists(gt):
            print("error path " + gt)
        gt_img = np.load(gt)
        #save_singledisp(gt_img, output_directory, VAL_IMAGES[i] + '_gt')
        error_img, error_score = get_scores_file_addtional(disparities[i, :, :], i, myresult)

        # error_img, error_score = get_scores_file_all(disparities[i, :, :], i, myresult)
        # save_erroplt_all(gt_img, disparities[i, :, :], error_img, output_directory, i, False)
        save_erroplt_additional(gt_img, disparities[i, :, :], error_img, output_directory, i, False)
        avg_score += error_score
        # get_scores(disparities[i, :, :],i)
        print("pp result")
        misc.imsave(output_directory + '/' + ADDITIONAL_IMAGES_ALL[i] + 'dispp.jpg', disparities_pp[i, :, :])
        error_img, error_score = get_scores_file_addtional(disparities_pp[i, :, :], i, myresult)

        save_singledisp_error(error_img, output_directory, ADDITIONAL_IMAGES_ALL[i] + 'errorplt')
        avg_score_pp +=error_score
        save_erroplt(gt_img, disparities_pp[i, :, :], error_img, output_directory, i, True)

        # get_scores(disparities_pp[i, :, :], i)
    print("-----------------avg score {:.5f}---------------".format(100.-(avg_score) / 8))
    myresult.write("-----------------avg score {}---------------".format(str(100.-(avg_score) / 8))+"\n")

    print("-----------------avg score {:.5f}---------------".format(100. - (avg_score_pp) / 8))
    myresult.write("-----------------avg score {}---------------".format(str(100. - (avg_score_pp) / 8)) + "\n")

    print('done.')
def test_MSE(params):
    """Test function."""
    dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.mode)
    center_image = [dataloader.center_image_batch]
    model = MonodepthModel(params, args.mode, center_image ,
                           reuse_variables=None,model_index=None)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    for step in range(num_test_samples):
        disp = sess.run(model.disp_center_est)
        disparities[step] = disp[0].squeeze()
        disparities_pp[step] = post_process_disparity(disp.squeeze())
        origin_pfm_dir = os.path.join(pfmdir,"originpfm")
        pp_pfm_dir = os.path.join(pfmdir,"pppfm")
        if not os.path.exists(origin_pfm_dir):
            os.mkdir(origin_pfm_dir)
        if not os.path.exists(pp_pfm_dir):
            os.mkdir(pp_pfm_dir)
        write_pfm(disp[0, :, :, 0], origin_pfm_dir + '/%s.pfm' % RMSE_MAE_BPR_IMAGES[step])
        write_pfm(disparities_pp[step], pp_pfm_dir + '/%s.pfm' % RMSE_MAE_BPR_IMAGES[step])
        misc.imsave(pp_pfm_dir + '/' + RMSE_MAE_BPR_IMAGES[step] + '_submit.jpg', disparities_pp[step])


    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy',    disparities)
    np.save(output_directory + '/disparities_pp.npy', disparities_pp)
    # misc.imsave('./' + 'disparities.png',disparities)
    # misc.imsave('./' + 'disparities_pp.png', disparities_pp)
    # for i in range(disparities.shape[0]):
    #     print ("origin result")
    #     myresult = open(output_directory + '/result.txt','a+')
    #     misc.imsave(output_directory+'/' + str(i) + 'dis.jpg', disparities[i, :, :])
    #     get_scores_file(disparities[i, :, :],i,myresult)
    #     # get_scores(disparities[i, :, :],i)
    #     print("pp result")
    #     misc.imsave(output_directory+'/' + str(i) + 'dispp.jpg', disparities_pp[i, :, :])
    #     get_scores_file(disparities_pp[i, :, :], i,myresult)
        # get_scores(disparities_pp[i, :, :], i)


    avg_score = 0.
    avg_score_pp = 0.
    myresult = open(output_directory + '/result.txt', 'a+')
    print("-----------------load checkpoint {}---------------".format(args.checkpoint_path))
    myresult.write("-----------------load checkpoint {}".format(args.checkpoint_path+"\n"))

    for i in range(disparities.shape[0]):
        print("origin result")

        if i <=7:
            misc.imsave(output_directory + '/' + RMSE_MAE_BPR_IMAGES[i] + '_disp.jpg', disparities[i, :, :])
            save_singledisp(disparities[i, :, :],output_directory,RMSE_MAE_BPR_IMAGES[i]+'plt')

        else:
            misc.imsave(output_directory + '/' + TEST_IMAGES[i] + '_disp.jpg', disparities[i, :, :])
            save_singledisp(disparities[i, :, :], output_directory, TEST_IMAGES[i]+'plt')
            continue
        # misc.imsave(output_directory + '/' + VAL_IMAGES_ALL[i] + '_dis.jpg', disparities[i, :, :])
        # get_scores_file(disparities[i, :, :],i,myresult)
        gt = os.path.join(gt_path, RMSE_MAE_BPR_IMAGES[i] + "/valid_disp_map.npy")
        # gt = os.path.join(gt_path, VAL_IMAGES_ALL[i] + "/valid_disp_map.npy")
        if not os.path.exists(gt):
            print("error path " + gt)
        gt_img = np.load(gt)
        #save_singledisp(gt_img, output_directory, VAL_IMAGES[i] + '_gt')
        error_img, error_score = get_scores_file_RMSE_MAE_BPR(disparities[i, :, :], i, myresult)
        # error_img, error_score = get_scores_file_all(disparities[i, :, :], i, myresult)
        # save_erroplt_all(gt_img, disparities[i, :, :], error_img, output_directory, i, False)
        save_erroplt(gt_img, disparities[i, :, :], error_img, output_directory, i, False)
        avg_score += error_score
        # get_scores(disparities[i, :, :],i)

        rmse_result,mae_result,bpr_result= cal_RMSE(disparities[i, :, :][10:-10,10:-10],gt_img)
        print("-----------------RMSE score {:.5f}---------------".format(rmse_result))
        myresult.write("-----------------RMSE score {}---------------".format(rmse_result) + "\n")
        print("-----------------MAE score {:.5f}---------------".format(mae_result))
        myresult.write("-----------------MAE score {}---------------".format(mae_result) + "\n")
        print("-----------------BPR score {:.5f}---------------".format(bpr_result))
        myresult.write("-----------------BPR score {}---------------".format(bpr_result) + "\n")


        print("pp result")
        misc.imsave(output_directory + '/' + VAL_IMAGES[i] + 'dispp.jpg', disparities_pp[i, :, :])
        error_img, error_score = get_scores_file_RMSE_MAE_BPR(disparities_pp[i, :, :], i, myresult)

        rmse_result_pp, mae_result_pp, bpr_result_pp = cal_RMSE(disparities_pp[i, :, :][10:-10, 10:-10], gt_img)
        print("-----------------RMSE score {:.5f}---------------".format(rmse_result_pp))
        myresult.write("-----------------RMSE score {}---------------".format(rmse_result_pp) + "\n")
        print("-----------------MAE score {:.5f}---------------".format(mae_result_pp))
        myresult.write("-----------------MAE score {}---------------".format(mae_result_pp) + "\n")
        print("-----------------BPR score {:.5f}---------------".format(bpr_result_pp))
        myresult.write("-----------------BPR score {}---------------".format(bpr_result_pp) + "\n")

        save_singledisp_error(error_img, output_directory, VAL_IMAGES[i] + 'errorplt')
        avg_score_pp +=error_score
        save_erroplt(gt_img, disparities_pp[i, :, :], error_img, output_directory, i, True)

        # get_scores(disparities_pp[i, :, :], i)
    print("-----------------avg score {:.5f}---------------".format(100.-(avg_score) / 8))
    myresult.write("-----------------avg score {}---------------".format(str(100.-(avg_score) / 8))+"\n")

    print("-----------------avg score {:.5f}---------------".format(100. - (avg_score_pp) / 8))
    myresult.write("-----------------avg score {}---------------".format(str(100. - (avg_score_pp) / 8)) + "\n")

    print('done.')
def main(_):

    params = monodepth_parameters(
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        dp_consistency_sigmoid_scale=args.dp_consistency_sigmoid_scale,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        centerSymmetry_loss_weight=args.centerSymmetry_loss_weight,
        disp_consistency_loss_weight=args.disp_consistency_loss_weight,
        full_summary=args.full_summary)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)
if __name__ == '__main__':
    tf.app.run()
