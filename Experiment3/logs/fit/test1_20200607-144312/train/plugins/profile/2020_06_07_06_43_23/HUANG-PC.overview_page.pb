�	O��DTy@O��DTy@!O��DTy@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$O��DTy@�}��1@1�:��Tkv@I������<@*	������z@2Y
"Iterator::Model::Prefetch::BatchV2�$��C�?!s=~�-U@)��3��?1�͙z��B@:Preprocessing2b
+Iterator::Model::Prefetch::BatchV2::Shuffle ۊ�e���?!�ၕ�G@):��H��?1ÿ́�v;@:Preprocessing2o
8Iterator::Model::Prefetch::BatchV2::Shuffle::TensorSlice {�/L�
�?!�c=i�4@){�/L�
�?1�c=i�4@:Preprocessing2F
Iterator::Model	�^)˰?!Ug\�.@)W[��재?1q����*@:Preprocessing2P
Iterator::Model::Prefetch�j+��݃?!�{�ɘ@)�j+��݃?1�{�ɘ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?7.1 % of the total step time sampled is spent on Kernel Launch.*moderate2A4.4 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�}��1@�}��1@!�}��1@      ��!       "	�:��Tkv@�:��Tkv@!�:��Tkv@*      ��!       2      ��!       :	������<@������<@!������<@B      ��!       J      ��!       R      ��!       Z      ��!       JGPU�"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMul/MatMulMatMul&|ue�7�?!&|ue�7�?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMul/MatMulMatMulFW���	�?!�i�/� �?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMulMatMul� Ja�?!�*p��?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._4/ffn/lin1/Tensordot/MatMul/MatMulMatMul�����ч?!��L�*�?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMulMatMulI�;a�̇?!����c��?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMul/MatMulMatMulB�@��ȇ?!��u����?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMul/MatMulMatMul}_d��?!~e�n�?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._3/ffn/lin1/Tensordot/MatMul/MatMulMatMul���$��?!=��.׷?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMul/MatMulMatMulL�M�u�?![�<�ź?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._0/ffn/lin1/Tensordot/MatMul/MatMulMatMul�	@ye�?!B\]���?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate?7.1 % of the total step time sampled is spent on Kernel Launch.moderate"A4.4 % of the total step time sampled is spent on All Others time.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 