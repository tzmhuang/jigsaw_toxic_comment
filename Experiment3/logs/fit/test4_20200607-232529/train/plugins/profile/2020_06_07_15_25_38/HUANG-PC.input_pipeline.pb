	�����z@�����z@!�����z@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�����z@
ףp=�1@1���=��w@I�QG��@@*	33333'�@2Y
"Iterator::Model::Prefetch::BatchV2��C�l�?!��y�T@)�/L�
F�?1�}���@@:Preprocessing2b
+Iterator::Model::Prefetch::BatchV2::Shuffle Dio����?!��ClV_I@)�j+����?1$��9p>@:Preprocessing2o
8Iterator::Model::Prefetch::BatchV2::Shuffle::TensorSlice B�f��j�?!?n��<�4@)B�f��j�?1?n��<�4@:Preprocessing2F
Iterator::Model�e��a��?!8鋧1@)ףp=
׳?1��c~��-@:Preprocessing2P
Iterator::Model::Prefetch46<�R�?!���B�� @)46<�R�?1���B�� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?7.6 % of the total step time sampled is spent on Kernel Launch.*moderate2A4.1 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	
ףp=�1@
ףp=�1@!
ףp=�1@      ��!       "	���=��w@���=��w@!���=��w@*      ��!       2      ��!       :	�QG��@@�QG��@@!�QG��@@B      ��!       J      ��!       R      ��!       Z      ��!       JGPU