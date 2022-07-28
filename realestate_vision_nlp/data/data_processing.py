
import tensorflow as tf
from tfrecord_helper import TFRecordHelper, TFRecordHelperWriter 

def source_from_image_tagging(bigstack_tfrecord):
  spec = TFRecordHelper.element_spec(tf.data.TFRecordDataset(bigstack_tfrecord), return_keys_only=True)
  print(spec)

  features = {
    'filenames': TFRecordHelper.DataType.VAR_STRING_ARRAY,
    'image_raw': TFRecordHelper.DataType.STRING,
    'orig_aspect_ratios': TFRecordHelper.DataType.VAR_FLOAT_ARRAY
  }