from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union
from enum import Enum
from pathlib import Path

import re, os, tarfile, shutil
import numpy as np
from realestate_core.common.run_config import home, bOnColab, bOnKaggle

from tfrecord_helper import TFRecordHelper, TFRecordHelperWriter 

class DataSrc(Enum):
    GDRIVE = 1
    GCS = 2

def download_from_gdrive(file_id: str, dest_dir: str = None, debug=False):
    raise NotImplementedError("download_from_gdrive not implemented")

def get_image_features(tar_filename: str, 
                       use_tfds: bool = False, 
                       data_src: DataSrc = DataSrc.GDRIVE, 
                       file_url: Optional[str] = None, 
                       dest_dir: Path = Path('.'), 
                       batch_size=128, 
                       cleanup=True,
                       skip_download=False):
    
    ext = Path(tar_filename).suffix
    prefix = re.compile(r'(.*?).tar.*$').match(tar_filename).group(1)
    print(f'prefix: {prefix}, ext: {ext}')

    photos_dir = Path('photos')
    photos_dir.mkdir(exist_ok=True)

    if not skip_download:
        photos = photos_dir.lf('*.jpg')
        assert len(photos) == 0

        if bOnColab:
            if data_src == DataSrc.GDRIVE:
                tar_filepath = home/'ConditionSentiment'/'tmp'/tar_filename
            else:
                # !gsutil cp gs://ai-tests/tmp/{tar_filename} .      
                os.system(f'gsutil cp gs://ai-tests/tmp/{tar_filename} .')
                tar_filepath = tar_filename
        elif bOnKaggle:
            if data_src == DataSrc.GDRIVE:
                file_id = re.compile(r'.*\/d\/(.*?)\/view\?usp=sharing$').match(file_url).group(1)
                print(f'file_id: {file_id}')
                # !gdown {file_id}
                os.system(f'gdown {file_id}')
                shutil.move(tar_filename, '/tmp')      
            else:
                download_from_gcs(f'tmp/{tar_filename}')    
                shutil.move(tar_filename, '/tmp')

            tar_filepath = '/tmp/' + tar_filename

        else:  
            assert False

        if ext == '.gz':
            with tarfile.open(tar_filepath, 'r:gz') as f:
                f.extractall(photos_dir)
        else:
            with tarfile.open(tar_filepath, 'r') as f:
                f.extractall(photos_dir)

    photos = photos_dir.lf('*.jpg')
    print(f'len(photos): {len(photos)}')

    if use_tfds:
        # create tfds from images in photos_dir 
        file_ds = tf.data.Dataset.from_tensor_slices(
            {
                'filename': [f.name for f in photos_dir.lf('*.jpg')],
                'filepath': [str(f) for f in photos_dir.lf('*.jpg')]
            }
        )  

        def read_decode_resize_encode(filepath):
            img = tf.image.decode_jpeg(tf.io.read_file(filepath), channels=3)
            img = tf.image.resize(img, [224, 224], tf.image.ResizeMethod.BICUBIC)
            img = tf.cast(img, tf.uint8)
            return tf.image.encode_jpeg(img)

        data_ds = file_ds.map(lambda x: {'filename': x['filename'], 'image_raw': read_decode_resize_encode(x['filepath'])})

        features = {
          'filename': TFRecordHelper.DataType.STRING,
          'image_raw': TFRecordHelper.DataType.STRING,   # bytes for the encoded jpeg, png, etc.
        }

        with TFRecordHelperWriter(f'{prefix}.tfrecords', features = features) as f:
            f.write(data_ds)
            
        def rescale(x):
            return tf.cast(tf.image.decode_jpeg(x['image_raw'], channels=3), tf.float32)/255., x['filename']
            
        parse_fn = TFRecordHelper.parse_fn(features)
        img_ds = tf.data.TFRecordDataset(f'{prefix}.tfrecords').map(parse_fn, num_parallel_calls=AUTO).map(rescale, num_parallel_calls=AUTO)
        
        img_names_list, image_features = flax_clip.get_image_features(ds=img_ds, batch_size=batch_size)        
    else:
        img_names_list, image_features = flax_clip.get_image_features(photos, batch_size=batch_size)
    
    np.savez_compressed(dest_dir/f'clip_image_features_{prefix}', files=img_names_list, image_features=image_features)

    if cleanup:
        print("cleaning up artifacts")
        try:
            flax_clip.cleanup(photos, prefix)
        except:
            pass

        photos = photos_dir.lf('*.jpg')
        assert len(photos) == 0

