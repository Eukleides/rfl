import tensorflow as tf

def tf_memory(frac = 0.6):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = frac
    sess = tf.Session(config=config)
