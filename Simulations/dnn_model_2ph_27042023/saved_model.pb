��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
�
ConstConst*
_output_shapes

: *
dtype0*�
value�B� "�_�B[�cAG�A���A���Bh��B��B��@#B�PPB�A�hB��A s
@D{DA|v�A	�BA1R2@��Az�A��HAI�@'N�@7A�nA�+A�ڕ@��^@���@���AM��A��A
�
Const_1Const*
_output_shapes

: *
dtype0*�
value�B� "�$9�;q)�+�a�J*�>fx��m=����*?�@�?��	@��L@��*@���?�ti�]ڇ=a�$>2�u?1(�?�6?S��>턙=v">�c�>_Ƨ?8?A��?6�J?�Z?U��>�$x�$�>�(�>
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
~
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�N*&
shared_nameAdam/v/dense_1/kernel
�
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes
:	�N*
dtype0
�
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�N*&
shared_nameAdam/m/dense_1/kernel
�
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes
:	�N*
dtype0
{
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�N*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:�N*
dtype0
{
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�N*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:�N*
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �N*$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	 �N*
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �N*$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	 �N*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�N*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�N*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�N*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�N*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �N*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	 �N*
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0	
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
: *
dtype0
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
: *
dtype0
�
#serving_default_normalization_inputPlaceholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�
StatefulPartitionedCallStatefulPartitionedCall#serving_default_normalization_inputConst_1Constdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_21004

NoOpNoOp
�%
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*�%
value�%B�% B�%
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
5
0
1
2
3
4
$5
%6*
 
0
1
$2
%3*
* 
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
+trace_0
,trace_1
-trace_2
.trace_3* 
6
/trace_0
0trace_1
1trace_2
2trace_3* 
 
3	capture_0
4	capture_1* 
�
5
_variables
6_iterations
7_learning_rate
8_index_dict
9
_momentums
:_velocities
;_update_step_xla*

<serving_default* 
* 
* 
* 
* 
* 
RL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_15layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE*

=trace_0* 

0
1*

0
1*
* 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*

0
1
2*

L0*
* 
* 
 
3	capture_0
4	capture_1* 
 
3	capture_0
4	capture_1* 
 
3	capture_0
4	capture_1* 
 
3	capture_0
4	capture_1* 
 
3	capture_0
4	capture_1* 
 
3	capture_0
4	capture_1* 
 
3	capture_0
4	capture_1* 
 
3	capture_0
4	capture_1* 
* 
* 
C
60
M1
N2
O3
P4
Q5
R6
S7
T8*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
M0
O1
Q2
S3*
 
N0
P1
R2
T3*
* 
 
3	capture_0
4	capture_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
U	variables
V	keras_api
	Wtotal
	Xcount*
^X
VARIABLE_VALUEAdam/m/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/dense/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/dense/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*

W0
X1*

U	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemeanvariancecount_1dense/kernel
dense/biasdense_1/kerneldense_1/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biastotalcountConst_2* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_21309
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecount_1dense/kernel
dense/biasdense_1/kerneldense_1/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biastotalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_21376�
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_20831
normalization_input
normalization_sub_y
normalization_sqrt_x
dense_20809:	 �N
dense_20811:	�N 
dense_1_20825:	�N
dense_1_20827:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCallt
normalization/subSubnormalization_inputnormalization_sub_y*
T0*'
_output_shapes
:��������� Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

: \
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

: �
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:��������� �
dense/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_20809dense_20811*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_20808�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_20825dense_1_20827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_20824w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:$ 

_output_shapes

: :$ 

_output_shapes

: :e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
__inference__traced_save_21309
file_prefix)
read_disablecopyonread_mean: /
!read_1_disablecopyonread_variance: *
 read_2_disablecopyonread_count_1:	 8
%read_3_disablecopyonread_dense_kernel:	 �N2
#read_4_disablecopyonread_dense_bias:	�N:
'read_5_disablecopyonread_dense_1_kernel:	�N3
%read_6_disablecopyonread_dense_1_bias:,
"read_7_disablecopyonread_iteration:	 0
&read_8_disablecopyonread_learning_rate: ?
,read_9_disablecopyonread_adam_m_dense_kernel:	 �N@
-read_10_disablecopyonread_adam_v_dense_kernel:	 �N:
+read_11_disablecopyonread_adam_m_dense_bias:	�N:
+read_12_disablecopyonread_adam_v_dense_bias:	�NB
/read_13_disablecopyonread_adam_m_dense_1_kernel:	�NB
/read_14_disablecopyonread_adam_v_dense_1_kernel:	�N;
-read_15_disablecopyonread_adam_m_dense_1_bias:;
-read_16_disablecopyonread_adam_v_dense_1_bias:)
read_17_disablecopyonread_total: )
read_18_disablecopyonread_count: 
savev2_const_2
identity_39��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: m
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_mean"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOpread_disablecopyonread_mean^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
: u
Read_1/DisableCopyOnReadDisableCopyOnRead!read_1_disablecopyonread_variance"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp!read_1_disablecopyonread_variance^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_2/DisableCopyOnReadDisableCopyOnRead read_2_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp read_2_disablecopyonread_count_1^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0	*
_output_shapes
: y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_kernel^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 �N*
dtype0n

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 �Nd

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:	 �Nw
Read_4/DisableCopyOnReadDisableCopyOnRead#read_4_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp#read_4_disablecopyonread_dense_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�N*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�N`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:�N{
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_dense_1_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�N*
dtype0o
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Nf
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:	�Ny
Read_6/DisableCopyOnReadDisableCopyOnRead%read_6_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp%read_6_disablecopyonread_dense_1_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_7/DisableCopyOnReadDisableCopyOnRead"read_7_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp"read_7_disablecopyonread_iteration^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_learning_rate^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_9/DisableCopyOnReadDisableCopyOnRead,read_9_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp,read_9_disablecopyonread_adam_m_dense_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 �N*
dtype0o
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 �Nf
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	 �N�
Read_10/DisableCopyOnReadDisableCopyOnRead-read_10_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp-read_10_disablecopyonread_adam_v_dense_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 �N*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 �Nf
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	 �N�
Read_11/DisableCopyOnReadDisableCopyOnRead+read_11_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp+read_11_disablecopyonread_adam_m_dense_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�N*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�Nb
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�N�
Read_12/DisableCopyOnReadDisableCopyOnRead+read_12_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp+read_12_disablecopyonread_adam_v_dense_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�N*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�Nb
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:�N�
Read_13/DisableCopyOnReadDisableCopyOnRead/read_13_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp/read_13_disablecopyonread_adam_m_dense_1_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�N*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Nf
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	�N�
Read_14/DisableCopyOnReadDisableCopyOnRead/read_14_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp/read_14_disablecopyonread_adam_v_dense_1_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�N*
dtype0p
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�Nf
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	�N�
Read_15/DisableCopyOnReadDisableCopyOnRead-read_15_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp-read_15_disablecopyonread_adam_m_dense_1_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead-read_16_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp-read_16_disablecopyonread_adam_v_dense_1_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_17/DisableCopyOnReadDisableCopyOnReadread_17_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpread_17_disablecopyonread_total^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_18/DisableCopyOnReadDisableCopyOnReadread_18_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpread_18_disablecopyonread_count^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0savev2_const_2"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *"
dtypes
2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_38Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_39IdentityIdentity_38:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_39Identity_39:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
 __inference__wrapped_model_20786
normalization_input"
sequential_normalization_sub_y#
sequential_normalization_sqrt_xB
/sequential_dense_matmul_readvariableop_resource:	 �N?
0sequential_dense_biasadd_readvariableop_resource:	�ND
1sequential_dense_1_matmul_readvariableop_resource:	�N@
2sequential_dense_1_biasadd_readvariableop_resource:
identity��'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp�
sequential/normalization/subSubnormalization_inputsequential_normalization_sub_y*
T0*'
_output_shapes
:��������� o
sequential/normalization/SqrtSqrtsequential_normalization_sqrt_x*
T0*
_output_shapes

: g
"sequential/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
 sequential/normalization/MaximumMaximum!sequential/normalization/Sqrt:y:0+sequential/normalization/Maximum/y:output:0*
T0*
_output_shapes

: �
 sequential/normalization/truedivRealDiv sequential/normalization/sub:z:0$sequential/normalization/Maximum:z:0*
T0*'
_output_shapes
:��������� �
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	 �N*
dtype0�
sequential/dense/MatMulMatMul$sequential/normalization/truediv:z:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype0�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Ns
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������N�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�N*
dtype0�
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
IdentityIdentity#sequential/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:$ 

_output_shapes

: :$ 

_output_shapes

: :e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_20852
normalization_input
normalization_sub_y
normalization_sqrt_x
dense_20841:	 �N
dense_20843:	�N 
dense_1_20846:	�N
dense_1_20848:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCallt
normalization/subSubnormalization_inputnormalization_sub_y*
T0*'
_output_shapes
:��������� Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

: \
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

: �
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:��������� �
dense/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_20841dense_20843*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_20808�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_20846dense_1_20848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_20824w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:$ 

_output_shapes

: :$ 

_output_shapes

: :e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_20876

inputs
normalization_sub_y
normalization_sqrt_x
dense_20865:	 �N
dense_20867:	�N 
dense_1_20870:	�N
dense_1_20872:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCallg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:��������� Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

: \
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

: �
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:��������� �
dense/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_20865dense_20867*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_20808�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_20870dense_1_20872*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_20824w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:$ 

_output_shapes

: :$ 

_output_shapes

: :X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
%__inference_dense_layer_call_fn_21140

inputs
unknown:	 �N
	unknown_0:	�N
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_20808p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������N`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
B__inference_dense_1_layer_call_and_return_conditional_losses_21170

inputs1
matmul_readvariableop_resource:	�N-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�N*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������N: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�	
�
B__inference_dense_1_layer_call_and_return_conditional_losses_20824

inputs1
matmul_readvariableop_resource:	�N-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�N*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������N: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�	
�
*__inference_sequential_layer_call_fn_20929
normalization_input
unknown
	unknown_0
	unknown_1:	 �N
	unknown_2:	�N
	unknown_3:	�N
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_20914o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

: :$ 

_output_shapes

: :e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�

�
@__inference_dense_layer_call_and_return_conditional_losses_20808

inputs1
matmul_readvariableop_resource:	 �N.
biasadd_readvariableop_resource:	�N
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 �N*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Ns
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������NQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������Nb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������Nw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
*__inference_sequential_layer_call_fn_21083

inputs
unknown
	unknown_0
	unknown_1:	 �N
	unknown_2:	�N
	unknown_3:	�N
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_20914o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

: :$ 

_output_shapes

: :X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�'
�
__inference_adapt_step_21049
iterator%
add_readvariableop_resource:	 %
readvariableop_resource: '
readvariableop_2_resource: ��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�IteratorGetNext�ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�add/ReadVariableOp�
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:��������� *&
output_shapes
:��������� *
output_types
2h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*'
_output_shapes
:��������� l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 o
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	:��Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: K
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: I
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: �
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0*
validate_shape(�
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22$
AssignVariableOpAssignVariableOp2"
IteratorGetNextIteratorGetNext2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22 
ReadVariableOpReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator
�

�
@__inference_dense_layer_call_and_return_conditional_losses_21151

inputs1
matmul_readvariableop_resource:	 �N.
biasadd_readvariableop_resource:	�N
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	 �N*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Ns
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������NQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������Nb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������Nw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
#__inference_signature_wrapper_21004
normalization_input
unknown
	unknown_0
	unknown_1:	 �N
	unknown_2:	�N
	unknown_3:	�N
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_20786o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

: :$ 

_output_shapes

: :e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�R
�
!__inference__traced_restore_21376
file_prefix#
assignvariableop_mean: )
assignvariableop_1_variance: $
assignvariableop_2_count_1:	 2
assignvariableop_3_dense_kernel:	 �N,
assignvariableop_4_dense_bias:	�N4
!assignvariableop_5_dense_1_kernel:	�N-
assignvariableop_6_dense_1_bias:&
assignvariableop_7_iteration:	 *
 assignvariableop_8_learning_rate: 9
&assignvariableop_9_adam_m_dense_kernel:	 �N:
'assignvariableop_10_adam_v_dense_kernel:	 �N4
%assignvariableop_11_adam_m_dense_bias:	�N4
%assignvariableop_12_adam_v_dense_bias:	�N<
)assignvariableop_13_adam_m_dense_1_kernel:	�N<
)assignvariableop_14_adam_v_dense_1_kernel:	�N5
'assignvariableop_15_adam_m_dense_1_bias:5
'assignvariableop_16_adam_v_dense_1_bias:#
assignvariableop_17_total: #
assignvariableop_18_count: 
identity_20��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_count_1Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_1_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_1_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_iterationIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_learning_rateIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp&assignvariableop_9_adam_m_dense_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_v_dense_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp%assignvariableop_11_adam_m_dense_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp%assignvariableop_12_adam_v_dense_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_m_dense_1_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_v_dense_1_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_m_dense_1_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_v_dense_1_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
'__inference_dense_1_layer_call_fn_21160

inputs
unknown:	�N
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_20824o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������N: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�	
�
*__inference_sequential_layer_call_fn_20891
normalization_input
unknown
	unknown_0
	unknown_1:	 �N
	unknown_2:	�N
	unknown_3:	�N
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_20876o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

: :$ 

_output_shapes

: :e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_21107

inputs
normalization_sub_y
normalization_sqrt_x7
$dense_matmul_readvariableop_resource:	 �N4
%dense_biasadd_readvariableop_resource:	�N9
&dense_1_matmul_readvariableop_resource:	�N5
'dense_1_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOpg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:��������� Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

: \
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

: �
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:��������� �
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	 �N*
dtype0�
dense/MatMulMatMulnormalization/truediv:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������N�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�N*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:$ 

_output_shapes

: :$ 

_output_shapes

: :X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�	
�
*__inference_sequential_layer_call_fn_21066

inputs
unknown
	unknown_0
	unknown_1:	 �N
	unknown_2:	�N
	unknown_3:	�N
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_20876o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_output_shapes

: :$ 

_output_shapes

: :X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_20914

inputs
normalization_sub_y
normalization_sqrt_x
dense_20903:	 �N
dense_20905:	�N 
dense_1_20908:	�N
dense_1_20910:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCallg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:��������� Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

: \
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

: �
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:��������� �
dense/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_20903dense_20905*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_20808�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_20908dense_1_20910*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_20824w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:$ 

_output_shapes

: :$ 

_output_shapes

: :X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_21131

inputs
normalization_sub_y
normalization_sqrt_x7
$dense_matmul_readvariableop_resource:	 �N4
%dense_biasadd_readvariableop_resource:	�N9
&dense_1_matmul_readvariableop_resource:	�N5
'dense_1_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOpg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:��������� Y
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

: \
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

: �
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:��������� �
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	 �N*
dtype0�
dense/MatMulMatMulnormalization/truediv:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������N�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�N*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:������������������: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:$ 

_output_shapes

: :$ 

_output_shapes

: :X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
\
normalization_inputE
%serving_default_normalization_input:0������������������;
dense_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�i
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
Q
0
1
2
3
4
$5
%6"
trackable_list_wrapper
<
0
1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
�
+trace_0
,trace_1
-trace_2
.trace_32�
*__inference_sequential_layer_call_fn_20891
*__inference_sequential_layer_call_fn_20929
*__inference_sequential_layer_call_fn_21066
*__inference_sequential_layer_call_fn_21083�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z+trace_0z,trace_1z-trace_2z.trace_3
�
/trace_0
0trace_1
1trace_2
2trace_32�
E__inference_sequential_layer_call_and_return_conditional_losses_20831
E__inference_sequential_layer_call_and_return_conditional_losses_20852
E__inference_sequential_layer_call_and_return_conditional_losses_21107
E__inference_sequential_layer_call_and_return_conditional_losses_21131�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z/trace_0z0trace_1z1trace_2z2trace_3
�
3	capture_0
4	capture_1B�
 __inference__wrapped_model_20786normalization_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3	capture_0z4	capture_1
�
5
_variables
6_iterations
7_learning_rate
8_index_dict
9
_momentums
:_velocities
;_update_step_xla"
experimentalOptimizer
,
<serving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
: 2mean
: 2variance
:	 2count
�
=trace_02�
__inference_adapt_step_21049�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z=trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ctrace_02�
%__inference_dense_layer_call_fn_21140�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zCtrace_0
�
Dtrace_02�
@__inference_dense_layer_call_and_return_conditional_losses_21151�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zDtrace_0
:	 �N2dense/kernel
:�N2
dense/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
Jtrace_02�
'__inference_dense_1_layer_call_fn_21160�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0
�
Ktrace_02�
B__inference_dense_1_layer_call_and_return_conditional_losses_21170�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zKtrace_0
!:	�N2dense_1/kernel
:2dense_1/bias
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
L0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
3	capture_0
4	capture_1B�
*__inference_sequential_layer_call_fn_20891normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3	capture_0z4	capture_1
�
3	capture_0
4	capture_1B�
*__inference_sequential_layer_call_fn_20929normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3	capture_0z4	capture_1
�
3	capture_0
4	capture_1B�
*__inference_sequential_layer_call_fn_21066inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3	capture_0z4	capture_1
�
3	capture_0
4	capture_1B�
*__inference_sequential_layer_call_fn_21083inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3	capture_0z4	capture_1
�
3	capture_0
4	capture_1B�
E__inference_sequential_layer_call_and_return_conditional_losses_20831normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3	capture_0z4	capture_1
�
3	capture_0
4	capture_1B�
E__inference_sequential_layer_call_and_return_conditional_losses_20852normalization_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3	capture_0z4	capture_1
�
3	capture_0
4	capture_1B�
E__inference_sequential_layer_call_and_return_conditional_losses_21107inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3	capture_0z4	capture_1
�
3	capture_0
4	capture_1B�
E__inference_sequential_layer_call_and_return_conditional_losses_21131inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3	capture_0z4	capture_1
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
_
60
M1
N2
O3
P4
Q5
R6
S7
T8"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
<
M0
O1
Q2
S3"
trackable_list_wrapper
<
N0
P1
R2
T3"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�
3	capture_0
4	capture_1B�
#__inference_signature_wrapper_21004normalization_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z3	capture_0z4	capture_1
�B�
__inference_adapt_step_21049iterator"�
���
FullArgSpec
args�

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_dense_layer_call_fn_21140inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_dense_layer_call_and_return_conditional_losses_21151inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_1_layer_call_fn_21160inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_1_layer_call_and_return_conditional_losses_21170inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
U	variables
V	keras_api
	Wtotal
	Xcount"
_tf_keras_metric
$:"	 �N2Adam/m/dense/kernel
$:"	 �N2Adam/v/dense/kernel
:�N2Adam/m/dense/bias
:�N2Adam/v/dense/bias
&:$	�N2Adam/m/dense_1/kernel
&:$	�N2Adam/v/dense_1/kernel
:2Adam/m/dense_1/bias
:2Adam/v/dense_1/bias
.
W0
X1"
trackable_list_wrapper
-
U	variables"
_generic_user_object
:  (2total
:  (2count�
 __inference__wrapped_model_20786�34$%E�B
;�8
6�3
normalization_input������������������
� "1�.
,
dense_1!�
dense_1���������n
__inference_adapt_step_21049NC�@
9�6
4�1�
���������� IteratorSpec 
� "
 �
B__inference_dense_1_layer_call_and_return_conditional_losses_21170d$%0�-
&�#
!�
inputs����������N
� ",�)
"�
tensor_0���������
� �
'__inference_dense_1_layer_call_fn_21160Y$%0�-
&�#
!�
inputs����������N
� "!�
unknown����������
@__inference_dense_layer_call_and_return_conditional_losses_21151d/�,
%�"
 �
inputs��������� 
� "-�*
#� 
tensor_0����������N
� �
%__inference_dense_layer_call_fn_21140Y/�,
%�"
 �
inputs��������� 
� ""�
unknown����������N�
E__inference_sequential_layer_call_and_return_conditional_losses_20831�34$%M�J
C�@
6�3
normalization_input������������������
p

 
� ",�)
"�
tensor_0���������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_20852�34$%M�J
C�@
6�3
normalization_input������������������
p 

 
� ",�)
"�
tensor_0���������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_21107x34$%@�=
6�3
)�&
inputs������������������
p

 
� ",�)
"�
tensor_0���������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_21131x34$%@�=
6�3
)�&
inputs������������������
p 

 
� ",�)
"�
tensor_0���������
� �
*__inference_sequential_layer_call_fn_20891z34$%M�J
C�@
6�3
normalization_input������������������
p

 
� "!�
unknown����������
*__inference_sequential_layer_call_fn_20929z34$%M�J
C�@
6�3
normalization_input������������������
p 

 
� "!�
unknown����������
*__inference_sequential_layer_call_fn_21066m34$%@�=
6�3
)�&
inputs������������������
p

 
� "!�
unknown����������
*__inference_sequential_layer_call_fn_21083m34$%@�=
6�3
)�&
inputs������������������
p 

 
� "!�
unknown����������
#__inference_signature_wrapper_21004�34$%\�Y
� 
R�O
M
normalization_input6�3
normalization_input������������������"1�.
,
dense_1!�
dense_1���������