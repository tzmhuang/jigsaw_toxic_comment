�	��-�{z@��-�{z@!��-�{z@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��-�{z@��:���4@1&��|F@w@I#��u�>@*	     p{@2F
Iterator::Model�]K�=�?!�n�RA@)7�[ A�?1^��Dƴ>@:Preprocessing2Y
"Iterator::Model::Prefetch::BatchV2����z�?!�Ș�pP@)��?�߾�?1��0��[;@:Preprocessing2b
+Iterator::Model::Prefetch::BatchV2::Shuffle M�O���?!�xW�3C@)Zd;�O��?1F����4@:Preprocessing2o
8Iterator::Model::Prefetch::BatchV2::Shuffle::TensorSlice @�߾��?!����r1@)@�߾��?1����r1@:Preprocessing2P
Iterator::Model::PrefetchX9��v��?!��o��>@)X9��v��?1��o��>@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?7.3 % of the total step time sampled is spent on Kernel Launch.*moderate2A4.9 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��:���4@��:���4@!��:���4@      ��!       "	&��|F@w@&��|F@w@!&��|F@w@*      ��!       2      ��!       :	#��u�>@#��u�>@!#��u�>@B      ��!       J      ��!       R      ��!       Z      ��!       JGPU�"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._5/ffn/lin2/Tensordot/MatMul/MatMulMatMul_-#�>9�?!_-#�>9�?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._1/ffn/lin2/Tensordot/MatMul/MatMulMatMul�FX1��?!(���&�?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMul/MatMulMatMul\���g�?!�U�v"��?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._0/ffn/lin1/Tensordot/MatMul/MatMulMatMul�`�!��?!#��>H�?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._4/ffn/lin1/Tensordot/MatMul/MatMul_1MatMult��X���?!�j��:�?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._2/ffn/lin1/Tensordot/MatMul/MatMul_1MatMulh��?�?!�,H�+�?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._3/ffn/lin2/Tensordot/MatMul/MatMulMatMul?S�η�?!5W���?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._4/ffn/lin2/Tensordot/MatMul/MatMulMatMul�%i�?!u�_��ɹ?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._0/ffn/lin2/Tensordot/MatMul/MatMulMatMul�P���?!7�~��?"�
�gradient_tape/distilbert_hs_mean_max_min_dense/tf_distil_bert_model/distilbert/transformer/layer_._2/ffn/lin2/Tensordot/MatMul/MatMulMatMul?st���?!��8}��?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate?7.3 % of the total step time sampled is spent on Kernel Launch.moderate"A4.9 % of the total step time sampled is spent on All Others time.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 