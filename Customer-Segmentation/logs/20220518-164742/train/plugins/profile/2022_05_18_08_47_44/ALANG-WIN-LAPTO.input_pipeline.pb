	\???(?@\???(?@!\???(?@	h??ؾ??h??ؾ??!h??ؾ??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$\???(?@?@??ǘ??A????S@Ya2U0*???*	????̌P@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?0?*???!>saF?\>@)/n????1,X??
?:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatU???N@??!]??3f<@)???H??1????8@:Preprocessing2F
Iterator::Model??_?L??!??(?k?@)-C??6??1eWMT?U3@:Preprocessing2U
Iterator::Model::ParallelMapV2????Mb??!??`?O+(@)????Mb??1??`?O+(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?z6?>??!~???%Q@)HP?s?r?1?-/ih?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????g?!4f?̅@)?????g?14f?̅@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zd?!?ظ?#6@){?G?zd?1?ظ?#6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?I+???!wr?Ɲ@@)ŏ1w-!_?1????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9i??ؾ??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?@??ǘ???@??ǘ??!?@??ǘ??      ??!       "      ??!       *      ??!       2	????S@????S@!????S@:      ??!       B      ??!       J	a2U0*???a2U0*???!a2U0*???R      ??!       Z	a2U0*???a2U0*???!a2U0*???JCPU_ONLYYi??ؾ??b 