�	z4Փyoz@z4Փyoz@!z4Փyoz@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$z4Փyoz@����3@1Q�+�OVw@I�J���r>@*	33333�q@2Y
"Iterator::Model::Prefetch::BatchV2I.�!���?!��9��rT@)=
ףp=�?1��4�A@:Preprocessing2b
+Iterator::Model::Prefetch::BatchV2::Shuffle *��D��?!#�o���F@)r�����?1H7���8@:Preprocessing2o
8Iterator::Model::Prefetch::BatchV2::Shuffle::TensorSlice ŏ1w-!�?!���Eo>5@)ŏ1w-!�?1���Eo>5@:Preprocessing2F
Iterator::Model[B>�٬�?!���<42@)�H.�!��?17��l`/@:Preprocessing2P
Iterator::Model::Prefetch��H�}}?!ʝ��3 @)��H�}}?1ʝ��3 @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?7.2 % of the total step time sampled is spent on Kernel Launch.*moderate2A4.5 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����3@����3@!����3@      ��!       "	Q�+�OVw@Q�+�OVw@!Q�+�OVw@*      ��!       2      ��!       :	�J���r>@�J���r>@!�J���r>@B      ��!       J      ��!       R      ��!       Z      ��!       JGPU�"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMulMatMul�����ʋ?!�����ʋ?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMul/MatMulMatMulV���=�?!�[67]��?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._3/ffn/lin1/Tensordot/MatMul/MatMul_1MatMul�6�kY�?!f��6u�?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._4/ffn/lin1/Tensordot/MatMul/MatMul_1MatMul'!~�1�?!���]�?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._5/ffn/lin1/Tensordot/MatMul/MatMul_1MatMul�+���?!R��BP�?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMul/MatMulMatMul�ZR��?!K03�l�?"�
pdistilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._1/ffn/lin1/Tensordot/MatMulMatMulD���و?!l 3m��?"�
pdistilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMulMatMul]�Y�&��?!�:�I��?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMul/MatMulMatMulF���C�?!a0�̖��?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMulMatMul�S�e�͆?!��9U]�?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate?7.2 % of the total step time sampled is spent on Kernel Launch.moderate"A4.5 % of the total step time sampled is spent on All Others time.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 