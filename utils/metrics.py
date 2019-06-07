import tensorflow as tf
from scipy.stats import pearsonr, spearmanr


def matt_corr(y_true, y_pred):
    y_pred_pos = tf.keras.backend.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = tf.keras.backend.sum(y_pos * y_pred_pos)
    tn = tf.keras.backend.sum(y_neg * y_pred_neg)

    fp = tf.keras.backend.sum(y_neg * y_pred_pos)
    fn = tf.keras.backend.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = tf.keras.backend.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + tf.keras.backend.epsilon())


def f1_score(y_true, y_pred):
    y_pred_pos = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = tf.keras.backend.sum(y_pos * y_pred_pos)
    tn = tf.keras.backend.sum(y_neg * y_pred_neg)

    fp = tf.keras.backend.sum(y_neg * y_pred_pos)
    fn = tf.keras.backend.sum(y_pos * y_pred_neg)

    numerator = 2 * tp
    denominator = 2 * tp + fn + fp

    return numerator / (denominator + tf.keras.backend.K.epsilon())


# def pear_corr(y_true, y_pred):
#     def _p_corr(y_true, y_pred):
#         try:
#             return pearsonr(y_true, y_pred)[0]
#         except:
#             return 0

#     return tf.py_function(_p_corr, (y_true, y_pred), tf.double)


def pear_corr(y_true, y_pred):

    y_true -= tf.keras.backend.mean(y_true)
    y_pred -= tf.keras.backend.mean(y_pred)

    y_true = tf.keras.backend.l2_normalize(y_true)
    y_pred = tf.keras.backend.l2_normalize(y_pred)

    pearson_correlation = tf.keras.backend.sum(y_true * y_pred)
    return pearson_correlation


def spear_corr(y_true, y_pred):
    def _s_corr(y_true, y_pred):
        try:
            return spearmanr(y_true, y_pred)[0]
        except:
            return 0

    return tf.py_function(_s_corr, (y_true, y_pred), tf.double)
