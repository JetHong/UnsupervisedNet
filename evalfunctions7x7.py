import sys
import os

sys.path.insert(0, sys.path[0] + '/../evaluation_toolkit/source')
import evaluation_toolkit.source.toolkit.utils.misc as my_misc
import evaluation_toolkit.source.toolkit.settings as setting
from random import *
import eval_tools
from evaluate import *
from evaluation_toolkit.source.toolkit.metrics.general_metrics import BadPix
import matplotlib
from matplotlib import pyplot as plt

VAL_IMAGES = [
    # "boxes"
    "sideboard", "cotton", "boxes", "dino",
    # # "antinous","greek","dishes","tower",
    "backgammon", "pyramids", "stripes", "dots"
    # "rosemary","boardgames","museum","pillows","tomb","vinyl",
    # "kitchen","medieval2","pens","platonic","table","town",
    # 'buddha', 'buddha2', 'monasRoom', 'papillon', 'stillLife'
]
VAL_IMAGES_ALL = [
    "sideboard", "cotton", "boxes", "dino",
    "antinous", "greek", "dishes", "tower",
    "backgammon", "pyramids", "stripes", "dots",
    "rosemary", "boardgames", "museum", "pillows", "tomb", "vinyl",
    "kitchen", "medieval2", "pens", "platonic", "table", "town",
    # 'buddha', 'buddha2', 'monasRoom', 'papillon', 'stillLife'
]

ADDITIONAL_IMAGES_ALL = ['museum', 'platonic', 'dishes', 'pens',
                         'pillows', 'vinyl', 'tower', 'table',
                         'rosemary', 'boardgames', 'town', 'kitchen',
                         'greek', 'tomb', 'antinous', 'medieval2']

TEST_IMAGES = [
    "sideboard", "cotton", "boxes", "dino",
    # "antinous","greek","dishes","tower",
    "backgammon", "pyramids", "stripes", "dots",
    "bedroom", "bicycle", "herbs", "origami"
    # "rosemary","boardgames","museum","pillows","tomb","vinyl",
    # "kitchen","medieval2","pens","platonic","table","town",
    # 'buddha', 'buddha2', 'monasRoom', 'papillon', 'stillLife'
]
RMSE_MAE_BPR_IMAGES = ["kitchen", "pillows", "tomb", "vinyl"]
OLD_TEST_IMAGES = [
    "maria", "statue", "buddha2", "stillLife", "cube", "papillon", "couple", "buddha", "monasRoom", "pyramide"
]


def get_scores(img, nb):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)
    evaluator = Evaluator()
    category = my_misc.infer_scene_category(VAL_IMAGES[nb])
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "./../evaluation_toolkit/data"
    sceneEval = my_misc.get_scene(VAL_IMAGES[nb], category, data_path=EVAL_ROOT)
    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    print ("-----------------scene {}---------------").format(VAL_IMAGES[nb])
    print ("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    return pre_err


def error_params_for_plt(err, title=""):
    # params refer to p_utils.show_img(img, title="", norm=None, show_axes_ticks=True, with_colorbar=False)
    err_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    return (err, title, err_norm, False, True)


def get_scores_file(img, nb, myfile):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    if myfile is None:
        print ("please send into the file")
        return
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)  # 0.5
    category = my_misc.infer_scene_category(VAL_IMAGES[nb])
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "evaluation_toolkit/data/full_data"
    sceneEval = my_misc.get_scene(VAL_IMAGES[nb], category, data_path=EVAL_ROOT)
    # pre_err_new = eval_tools.compute_errorimage(sceneEval, [bp007], img,True)[1][0]
    pre_err_new = eval_tools.compute_scores(sceneEval, [bp007], img, True)[1][0]

    # fig = plt.figure(figsize=(24, 6))
    # # plt.figure(figsize=(6, 6.5))
    # ax = fig.add_subplot(131)
    # ax.set_title("result")
    # ax.imshow(img)#gt
    # ax = fig.add_subplot(132)
    # ax.set_title("error img")
    # ax.imshow(pre_err_new)
    # ax = fig.add_subplot(133)#result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1.,pre_err_new), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    # figname = VAL_IMAGES[nb]+".png"
    # fig.savefig('figname')

    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    # disp_gt = np.load("{}/{}.npy".format(f_dir, f_name_split))
    # disp_gt = disp_gt.astype(np.float32)
    # disp_err_pre, value_acc = evaluator.error_acc(img, disp_gt, eval_mask=None)
    print ("-----------------scene {}---------------".format(VAL_IMAGES[nb]))
    print ("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    myfile.write("-----------------scene {}---------------".format(VAL_IMAGES[nb] + '\n'))
    myfile.write(
        "Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err) + '\n')
    myfile.flush()
    return pre_err_new, pre_err


def get_scores_file_by_name(img, scene_name, myfile):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    if myfile is None:
        print ("please send into the file")
        return
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)  # 0.5
    category = my_misc.infer_scene_category(scene_name)
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "./../evaluation_toolkit/data"
    sceneEval = my_misc.get_scene(scene_name, category, data_path=EVAL_ROOT)
    pre_err_new = eval_tools.compute_errorimage(sceneEval, [bp007], img, True)[1][0]

    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    print ("-----------------scene {}---------------").format(scene_name)
    print ("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    myfile.write("-----------------scene {}---------------".format(scene_name + '\n'))
    myfile.write(
        "Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err) + '\n')
    myfile.flush()
    return pre_err_new, pre_err


def get_scores_file_addtional(img, nb, myfile):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    if myfile is None:
        print ("please send into the file")
        return
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)  # 0.5
    category = my_misc.infer_scene_category(ADDITIONAL_IMAGES_ALL[nb])
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "./../evaluation_toolkit/data"
    sceneEval = my_misc.get_scene(ADDITIONAL_IMAGES_ALL[nb], category, data_path=EVAL_ROOT)
    pre_err_new = eval_tools.compute_errorimage(sceneEval, [bp007], img, True)[1][0]

    # fig = plt.figure(figsize=(24, 6))
    # # plt.figure(figsize=(6, 6.5))
    # ax = fig.add_subplot(131)
    # ax.set_title("result")
    # ax.imshow(img)#gt
    # ax = fig.add_subplot(132)
    # ax.set_title("error img")
    # ax.imshow(pre_err_new)
    # ax = fig.add_subplot(133)#result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1.,pre_err_new), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    # figname = VAL_IMAGES[nb]+".png"
    # fig.savefig('figname')

    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    # disp_gt = np.load("{}/{}.npy".format(f_dir, f_name_split))
    # disp_gt = disp_gt.astype(np.float32)
    # disp_err_pre, value_acc = evaluator.error_acc(img, disp_gt, eval_mask=None)
    print ("-----------------scene {}---------------").format(ADDITIONAL_IMAGES_ALL[nb])
    print ("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    myfile.write("-----------------scene {}---------------".format(ADDITIONAL_IMAGES_ALL[nb] + '\n'))
    myfile.write(
        "Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err) + '\n')
    myfile.flush()
    return pre_err_new, pre_err


def get_scores_file_RMSE_MAE_BPR(img, nb, myfile):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    if myfile is None:
        print ("please send into the file")
        return
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)  # 0.5
    category = my_misc.infer_scene_category(RMSE_MAE_BPR_IMAGES[nb])
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "./../evaluation_toolkit/data"
    sceneEval = my_misc.get_scene(RMSE_MAE_BPR_IMAGES[nb], category, data_path=EVAL_ROOT)
    pre_err_new = eval_tools.compute_scores(sceneEval, [bp007], img, True)[1][0]

    # fig = plt.figure(figsize=(24, 6))
    # # plt.figure(figsize=(6, 6.5))
    # ax = fig.add_subplot(131)
    # ax.set_title("result")
    # ax.imshow(img)#gt
    # ax = fig.add_subplot(132)
    # ax.set_title("error img")
    # ax.imshow(pre_err_new)
    # ax = fig.add_subplot(133)#result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1.,pre_err_new), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    # figname = VAL_IMAGES[nb]+".png"
    # fig.savefig('figname')

    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    # disp_gt = np.load("{}/{}.npy".format(f_dir, f_name_split))
    # disp_gt = disp_gt.astype(np.float32)
    # disp_err_pre, value_acc = evaluator.error_acc(img, disp_gt, eval_mask=None)
    print ("-----------------scene {}---------------").format(RMSE_MAE_BPR_IMAGES[nb])
    print ("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    myfile.write("-----------------scene {}---------------".format(RMSE_MAE_BPR_IMAGES[nb] + '\n'))
    myfile.write(
        "Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err) + '\n')
    myfile.flush()
    return pre_err_new, pre_err


def get_scores_file_all(img, nb, myfile):
    """
    this funciton  gets the scores
    :param img: numpy shape[h,w]
    :param nb: sequence of the val img
    :return:score
    """
    if myfile is None:
        print ("please send into the file")
        return
    ERROR_RANGE = 0.07
    bp007 = BadPix(0.07)  # 0.5
    category = my_misc.infer_scene_category(VAL_IMAGES_ALL[nb])
    # Backgammon, Stripes will set gt_scale to 10 when compute_scores(), so leave it 1.0 here
    EVAL_ROOT = "./../evaluation_toolkit/data"
    sceneEval = my_misc.get_scene(VAL_IMAGES_ALL[nb], category, data_path=EVAL_ROOT)
    pre_err_new = eval_tools.compute_scores(sceneEval, [bp007], img, True)[1][0]

    # fig = plt.figure(figsize=(24, 6))
    # # plt.figure(figsize=(6, 6.5))
    # ax = fig.add_subplot(131)
    # ax.set_title("result")
    # ax.imshow(img)#gt
    # ax = fig.add_subplot(132)
    # ax.set_title("error img")
    # ax.imshow(pre_err_new)
    # ax = fig.add_subplot(133)#result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1.,pre_err_new), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    # figname = VAL_IMAGES[nb]+".png"
    # fig.savefig('figname')

    pre_err = eval_tools.compute_scores(sceneEval, [bp007], img)[bp007.get_id()]['value']
    # disp_gt = np.load("{}/{}.npy".format(f_dir, f_name_split))
    # disp_gt = disp_gt.astype(np.float32)
    # disp_err_pre, value_acc = evaluator.error_acc(img, disp_gt, eval_mask=None)
    print ("-----------------scene {}---------------").format(VAL_IMAGES_ALL[nb])
    print ("Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err))
    myfile.write("-----------------scene {}---------------".format(VAL_IMAGES_ALL[nb] + '\n'))
    myfile.write(
        "Value({}) accuracy:right rate {:.3f} error rate {:.3f}".format(ERROR_RANGE, 100 - pre_err, pre_err) + '\n')
    myfile.flush()
    return pre_err_new, pre_err


def save_erroplt_all(groundtruth, myresult, error_img, output_directory, image_number, is_pp):
    """
    save the plt into the output_directory
    :param myresult:
    :param error_img:
    :param output_directory:
    :return:
    """
    fig = plt.figure(figsize=(24, 6))
    # plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(141)
    ax.set_title("gt")
    ax.imshow(groundtruth)
    ax = fig.add_subplot(142)
    ax.set_title("result")
    ax.imshow(myresult)
    ax = fig.add_subplot(143)
    ax.set_title("error img")
    ax.imshow(error_img)
    # ax = fig.add_subplot(144)  # result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1., error_img), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    if is_pp:
        # fignamepath = os.path.join(output_directory,VAL_IMAGES[image_number]+"_pp_fig" + ".png")
        fignamepath = os.path.join(output_directory, VAL_IMAGES_ALL[image_number] + "_pp_fig" + ".png")
    else:
        # fignamepath = os.path.join(output_directory, VAL_IMAGES[image_number] + "fig.png")
        fignamepath = os.path.join(output_directory, VAL_IMAGES_ALL[image_number] + "fig.png")
    fig.savefig(fignamepath)
    plt.close()
    return 0


def save_erroplt(groundtruth, myresult, error_img, output_directory, image_number, is_pp):
    """
    save the plt into the output_directory
    :param myresult:
    :param error_img:
    :param output_directory:
    :return:
    """
    bp007 = BadPix(0.07)
    fig = plt.figure(figsize=(24, 6))
    # plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(141)
    ax.set_title("gt")
    ax.imshow(groundtruth)
    ax = fig.add_subplot(142)
    ax.set_title("result")
    ax.imshow(myresult)
    ax = fig.add_subplot(143)
    ax.set_title("error img")
    ax.imshow(error_img, **setting.metric_args(bp007))
    # ax = fig.add_subplot(144)  # result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1., error_img), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    if is_pp:
        fignamepath = os.path.join(output_directory, VAL_IMAGES[image_number] + "_pp_fig" + ".png")
    else:
        fignamepath = os.path.join(output_directory, VAL_IMAGES[image_number] + "fig.png")
    fig.savefig(fignamepath)
    plt.close()
    return 0


def save_erroplt_by_name(groundtruth, myresult, error_img, output_directory, scene_name, is_pp):
    """
    save the plt into the output_directory
    :param myresult:
    :param error_img:
    :param output_directory:
    :return:
    """
    bp007 = BadPix(0.07)
    fig = plt.figure(figsize=(24, 6))
    # plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(141)
    ax.set_title("gt")
    ax.imshow(groundtruth)
    ax = fig.add_subplot(142)
    ax.set_title("result")
    ax.imshow(myresult)
    ax = fig.add_subplot(143)
    ax.set_title("error img")
    ax.imshow(error_img, **setting.metric_args(bp007))
    # ax = fig.add_subplot(144)  # result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1., error_img), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    if is_pp:
        fignamepath = os.path.join(output_directory, scene_name + "_pp_fig" + ".png")
    else:
        fignamepath = os.path.join(output_directory, scene_name + "fig.png")
    fig.savefig(fignamepath)
    plt.close()
    return 0


def save_erroplt_additional(groundtruth, myresult, error_img, output_directory, image_number, is_pp):
    """
    save the plt into the output_directory
    :param myresult:
    :param error_img:
    :param output_directory:
    :return:
    """
    bp007 = BadPix(0.07)
    fig = plt.figure(figsize=(24, 6))
    # plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(141)
    ax.set_title("gt")
    ax.imshow(groundtruth)
    ax = fig.add_subplot(142)
    ax.set_title("result")
    ax.imshow(myresult)
    ax = fig.add_subplot(143)
    ax.set_title("error img")
    ax.imshow(error_img, **setting.metric_args(bp007))
    # ax = fig.add_subplot(144)  # result
    # ax.set_title("reder is error img")
    # im = ax.imshow(np.subtract(1., error_img), cmap=plt.cm.autumn)
    # plt.colorbar(im)
    if is_pp:
        fignamepath = os.path.join(output_directory, ADDITIONAL_IMAGES_ALL[image_number] + "_pp_fig" + ".png")
    else:
        fignamepath = os.path.join(output_directory, ADDITIONAL_IMAGES_ALL[image_number] + "fig.png")
    fig.savefig(fignamepath)
    plt.close()
    return 0

def write_pfm(data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write((file_identifier))
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())

        file.write(values)


def save_singleimg(data, fpath, fname):
    # fig = plt.figure(figsize=(5.12, 5.12))
    # plt.axis('off')
    # ax = fig.add_subplot(111)
    #
    # ax.imshow(data)
    # fignamepath = os.path.join(fpath, fname + "fig.png")
    # fig.savefig(fignamepath,bbox_inches='tight')
    # plt.close()

    fig, ax = plt.subplots()
    im = data
    ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width, channel = np.shape(im)
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    fignamepath = os.path.join(fpath, fname + ".png")
    plt.savefig(fignamepath, dpi=300)
    return


def save_singledisp(data, fpath, fname):
    # fig = plt.figure(figsize=(5.12, 5.12))
    # plt.axis('off')
    # ax = fig.add_subplot(111)
    #
    # ax.imshow(data)
    # fignamepath = os.path.join(fpath, fname + "fig.png")
    # fig.savefig(fignamepath,bbox_inches='tight')
    # plt.close()

    fig, ax = plt.subplots()
    im = data

    # bp007 = BadPix(0.07)
    # ax.imshow(im, aspect='equal',**setting.metric_args(bp007))
    ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width = np.shape(im)
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    fignamepath = os.path.join(fpath, fname + "fig.png")
    plt.savefig(fignamepath, dpi=300)
    plt.close()
    return


def save_singledisp_right(data, fpath, fname):
    # fig = plt.figure(figsize=(5.12, 5.12))
    # plt.axis('off')
    # ax = fig.add_subplot(111)
    #
    # ax.imshow(data)
    # fignamepath = os.path.join(fpath, fname + "fig.png")
    # fig.savefig(fignamepath,bbox_inches='tight')
    # plt.close()

    fig, ax = plt.subplots()
    im = data

    # bp007 = BadPix(0.07)
    # ax.imshow(im, aspect='equal',**setting.metric_args(bp007))
    ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width = np.shape(im)
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    fignamepath = os.path.join(fpath, fname + "fig.png")
    plt.savefig(fignamepath, dpi=300)
    plt.close()
    return


def save_singledisp_realword(data, fpath, fname, min, max):
    # fig = plt.figure(figsize=(5.12, 5.12))
    # plt.axis('off')
    # ax = fig.add_subplot(111)
    #
    # ax.imshow(data)
    # fignamepath = os.path.join(fpath, fname + "fig.png")
    # fig.savefig(fignamepath,bbox_inches='tight')
    # plt.close()
    #
    # fig, ax = plt.subplots()
    # im = data
    #
    # # bp007 = BadPix(0.07)
    # # ax.imshow(im, aspect='equal',**setting.metric_args(bp007))
    # ax.imshow(im, aspect='equal')
    # plt.axis('off')
    # height, width = np.shape(im)
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # fignamepath = os.path.join(fpath, fname + "fig.png")
    # plt.savefig(fignamepath, dpi = 300)
    # plt.close()

    img_root = "/home/sunshine77/PycharmProjects/monodepth_star/evaluation_toolkit/data"
    category = my_misc.infer_scene_category(VAL_IMAGES_ALL[0])
    scene = my_misc.get_scene(VAL_IMAGES_ALL[0], category, data_path=img_root)
    show_args = setting.disp_map_args(scene)
    show_args["vmax"] = max
    show_args["vmin"] = min
    fig, ax = plt.subplots()
    im = data
    # im = data[10:-10, 10:-10]
    # bp007 = BadPix(0.07)
    # ax.imshow(im, aspect='equal',**setting.metric_args(bp007))
    ax.imshow(im, aspect='equal', **show_args)
    plt.axis('off')
    height, width = np.shape(im)
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    fignamepath = os.path.join(fpath, fname + "fig.png")
    plt.savefig(fignamepath, dpi=300)

    return


def save_singledisp_error(data, fpath, fname):
    # fig = plt.figure(figsize=(5.12, 5.12))
    # plt.axis('off')
    # ax = fig.add_subplot(111)
    #
    # ax.imshow(data)
    # fignamepath = os.path.join(fpath, fname + "fig.png")
    # fig.savefig(fignamepath,bbox_inches='tight')
    # plt.close()

    fig, ax = plt.subplots()
    # im = data
    im = data[15:-15, 15:-15]

    bp007 = BadPix(0.07)
    ax.imshow(im, aspect='equal', **setting.metric_args(bp007))
    # ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width = np.shape(im)
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    fignamepath = os.path.join(fpath, fname + "fig.png")
    plt.savefig(fignamepath, dpi=300)
    plt.close()
    return


def cal_RMSE(disp, gt):
    rmse_pow = np.power(np.subtract(disp, gt), 2)
    rmse_result = np.sqrt(np.mean(rmse_pow))

    mae_result = np.mean(np.abs(np.subtract(disp, gt)))

    bpr_abs = np.abs(np.subtract(disp, gt))
    bpr = np.where(bpr_abs > 0.2, 1, 0)
    bpr_result = np.mean(bpr)

    return rmse_result, mae_result, bpr_result


def save_singledisp_val(data, fpath, fname):
    # fig = plt.figure(figsize=(5.12, 5.12))
    # plt.axis('off')
    # ax = fig.add_subplot(111)
    #
    # ax.imshow(data)
    # fignamepath = os.path.join(fpath, fname + "fig.png")
    # fig.savefig(fignamepath,bbox_inches='tight')
    # plt.close()
    img_root = "/home/sunshine77/PycharmProjects/monodepth_star/evaluation_toolkit/data"
    category = my_misc.infer_scene_category(fname)
    scene = my_misc.get_scene(fname, category, data_path=img_root)
    show_args = setting.disp_map_args(scene)

    fig, ax = plt.subplots()
    im = data
    # im = data[10:-10, 10:-10]
    # bp007 = BadPix(0.07)
    # ax.imshow(im, aspect='equal',**setting.metric_args(bp007))
    ax.imshow(im, aspect='equal', **show_args)
    plt.axis('off')
    height, width = np.shape(im)
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    fignamepath = os.path.join(fpath, fname + "fig.png")
    plt.savefig(fignamepath, dpi=300)
    return
