x
š
8
Const
output"dtype"
valuetensor"
dtypetype
Ą
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
StringLower	
input

output"
encodingstring "serve*2.13.12v2.13.0-17-gf841394b1b78ßQ
^
ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"               
b
Const_1Const*
_output_shapes
:*
dtype0*'
valueBBnegativeBpositive


hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_e347217f-27d6-49a4-8b2b-bb6234a6f225*
value_dtype0	
y
serving_default_inputsPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
{
serving_default_inputs_1Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
Ý
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsserving_default_inputs_1
hash_table*
Tin
2*
Tout
2	*
_collective_manager_ids
 *+
_output_shapes
:˙˙˙˙˙˙˙˙˙:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1496655
Ę
StatefulPartitionedCall_1StatefulPartitionedCall
hash_tableConst_1Const*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__initializer_1496666
(
NoOpNoOp^StatefulPartitionedCall_1
Ü
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B

created_variables
	resources
trackable_objects
initializers

assets
transform_fn

signatures* 
* 
	
0* 
* 
	
	0* 
* 
* 


serving_default* 
R
	_initializer
_create_resource
_initialize
_destroy_resource* 
* 
* 

trace_0* 

trace_0* 

trace_0* 
* 
 
	capture_1
	capture_2* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConst_2*
Tin
2*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1496698

StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename*
Tin
2*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1496707˝>


__inference_pruned_1496644

inputs
inputs_13
/key_value_init_lookuptableimportv2_table_handle
identity

identity_1	R
Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙_
keysConst*
_output_shapes
:*
dtype0*'
valueBBnegativeBpositive^
ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"               Q
inputs_copyIdentityinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
StringLowerStringLowerinputs_copy:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ţ
"key_value_init/LookupTableImportV2LookupTableImportV2/key_value_init_lookuptableimportv2_table_handlekeys:output:0Const:output:0*	
Tin0*

Tout0	*&
 _has_manual_control_dependencies(*
_output_shapes
 
None_Lookup/LookupTableFindV2LookupTableFindV2/key_value_init_lookuptableimportv2_table_handleinputs_1_copy:output:0Const_1:output:0#^key_value_init/LookupTableImportV2*	
Tin0*

Tout0	*&
 _has_manual_control_dependencies(*
_output_shapes
:
NoOpNoOp^None_Lookup/LookupTableFindV2#^key_value_init/LookupTableImportV2*&
 _has_manual_control_dependencies(*
_output_shapes
 c
IdentityIdentityStringLower:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w

Identity_1Identity&None_Lookup/LookupTableFindV2:values:0^NoOp*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ř
<
__inference__creator_1496659
identity˘
hash_table

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_e347217f-27d6-49a4-8b2b-bb6234a6f225*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
Ü

%__inference_signature_wrapper_1496655

inputs
inputs_1
unknown
identity

identity_1	˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1unknown*
Tin
2*
Tout
2	*+
_output_shapes
:˙˙˙˙˙˙˙˙˙:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *#
fR
__inference_pruned_1496644o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0	*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1:'#
!
_user_specified_name	1496649

I
#__inference__traced_restore_1496707
file_prefix

identity_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B Ł
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
2Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

o
 __inference__traced_save_1496698
file_prefix
savev2_const_2

identity_1˘MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B Ú
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const_2"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 7
NoOpNoOp^MergeV2Checkpoints*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:?;

_output_shapes
: 
!
_user_specified_name	Const_2
Ą
č
 __inference__initializer_14966663
/key_value_init_lookuptableimportv2_table_handle+
'key_value_init_lookuptableimportv2_keys,
(key_value_init_lookuptableimportv2_const	
identity˘"key_value_init/LookupTableImportV2ę
"key_value_init/LookupTableImportV2LookupTableImportV2/key_value_init_lookuptableimportv2_table_handle'key_value_init_lookuptableimportv2_keys(key_value_init_lookuptableimportv2_const*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: G
NoOpNoOp#^key_value_init/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2H
"key_value_init/LookupTableImportV2"key_value_init/LookupTableImportV2:, (
&
_user_specified_nametable_handle:@<

_output_shapes
:

_user_specified_namekeys:A=

_output_shapes
:

_user_specified_nameConst

.
__inference__destroyer_1496670
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes "ĘL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
9
inputs/
serving_default_inputs:0˙˙˙˙˙˙˙˙˙
=
inputs_11
serving_default_inputs_1:0˙˙˙˙˙˙˙˙˙=
	review_xf0
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙1
sentiment_xf!
StatefulPartitionedCall:1	tensorflow/serving/predict:

created_variables
	resources
trackable_objects
initializers

assets
transform_fn

signatures"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
2B0
__inference_pruned_1496644inputsinputs_1
,

serving_default"
signature_map
f
	_initializer
_create_resource
_initialize
_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
ÓBĐ
%__inference_signature_wrapper_1496655inputsinputs_1"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Í
trace_02°
__inference__creator_1496659
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ ztrace_0
Ń
trace_02´
 __inference__initializer_1496666
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ ztrace_0
Ď
trace_02˛
__inference__destroyer_1496670
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ ztrace_0
łB°
__inference__creator_1496659"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
ó
	capture_1
	capture_2B´
 __inference__initializer_1496666"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ z	capture_1z	capture_2
ľB˛
__inference__destroyer_1496670"
˛
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *˘ 
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstantA
__inference__creator_1496659!˘

˘ 
Ş "
unknown C
__inference__destroyer_1496670!˘

˘ 
Ş "
unknown J
 __inference__initializer_1496666&˘

˘ 
Ş "
unknown 
__inference_pruned_1496644ň~˘{
t˘q
oŞl
1
review'$
inputs_review˙˙˙˙˙˙˙˙˙
7
	sentiment*'
inputs_sentiment˙˙˙˙˙˙˙˙˙
Ş "mŞj
0
	review_xf# 
	review_xf˙˙˙˙˙˙˙˙˙
6
sentiment_xf&#
sentiment_xf˙˙˙˙˙˙˙˙˙	ř
%__inference_signature_wrapper_1496655Îi˘f
˘ 
_Ş\
*
inputs 
inputs˙˙˙˙˙˙˙˙˙
.
inputs_1"
inputs_1˙˙˙˙˙˙˙˙˙"^Ş[
0
	review_xf# 
	review_xf˙˙˙˙˙˙˙˙˙
'
sentiment_xf
sentiment_xf	