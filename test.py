from transformers import TFElectraForPreTraining
import tensorflow as tf

test = TFElectraForPreTraining.from_pretrained("./model_pretrained/dis/")

test(tf.constant([[64002,
                   8806,
                   2768,
                   1168,
                   2673,
                   2545,
                   1125,
                   1705,
                   2211,
                   4973,
                   1288,
                   1433,
                   64003]]))
