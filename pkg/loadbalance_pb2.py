# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: loadbalance.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11loadbalance.proto\x12\x0f\x44NNLoadBalancer\"|\n\x05Query\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x10\n\x08model_id\x18\x02 \x01(\x05\x12\n\n\x02\x62s\x18\x03 \x01(\x05\x12\x0f\n\x07seq_len\x18\x04 \x01(\x05\x12\x13\n\x0bstart_stamp\x18\x05 \x01(\x01\x12\x12\n\nqos_target\x18\x06 \x01(\x05\x12\x0f\n\x07load_id\x18\x07 \x01(\x05\"<\n\x06Result\x12\x0f\n\x07node_id\x18\x01 \x01(\x05\x12\x10\n\x08\x61\x63\x63\x65pted\x18\x02 \x01(\x08\x12\x0f\n\x07\x65lapsed\x18\x03 \x01(\x01\"\"\n\x05Layer\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0b\n\x03num\x18\x02 \x01(\x05\"0\n\x06Layers\x12&\n\x06layers\x18\x01 \x03(\x0b\x32\x16.DNNLoadBalancer.Layer\"I\n\x10GetLayersRequest\x12\r\n\x05start\x18\x01 \x01(\x05\x12\x0b\n\x03\x65nd\x18\x02 \x01(\x05\x12\r\n\x05model\x18\x03 \x01(\t\x12\n\n\x02\x62s\x18\x04 \x01(\x05\"[\n\x11GetLayersResponse\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\'\n\x06layers\x18\x03 \x03(\x0b\x32\x17.DNNLoadBalancer.Layers\"|\n\x14GetPredictionRequest\x12\x31\n\x06model1\x18\x01 \x01(\x0b\x32!.DNNLoadBalancer.GetLayersRequest\x12\x31\n\x06model2\x18\x02 \x01(\x0b\x32!.DNNLoadBalancer.GetLayersRequest\"Z\n\x15GetPredictionResponse\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\x12\n\nprediction\x18\x03 \x01(\x02\x12\x0e\n\x06\x61\x63tual\x18\x04 \x01(\x02\"b\n\x11GetResultsRequest\x12\x0e\n\x06model1\x18\x01 \x01(\t\x12\x0e\n\x06model2\x18\x02 \x01(\t\x12\x0c\n\x04load\x18\x03 \x01(\x02\x12\x10\n\x08\x64\x65\x61\x64line\x18\x04 \x01(\x02\x12\r\n\x05query\x18\x05 \x01(\x02\"\x1e\n\x0cListOfFloats\x12\x0e\n\x06metric\x18\x01 \x03(\x02\"\xc1\x01\n\x12GetResultsResponse\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x0f\n\x07message\x18\x02 \x01(\t\x12\x12\n\nviolation1\x18\x03 \x03(\x02\x12\x12\n\nviolation2\x18\x04 \x03(\x02\x12\x13\n\x0bthroughout1\x18\x05 \x03(\x02\x12\x13\n\x0bthroughout2\x18\x06 \x03(\x02\x12\r\n\x05tail1\x18\x07 \x03(\x02\x12\r\n\x05tail2\x18\x08 \x03(\x02\x12\r\n\x05mean1\x18\t \x03(\x02\x12\r\n\x05mean2\x18\n \x03(\x02\x32\xdc\x02\n\x0f\x44NNLoadBalancer\x12>\n\x0bLBInference\x12\x16.DNNLoadBalancer.Query\x1a\x17.DNNLoadBalancer.Result\x12R\n\tGetLayers\x12!.DNNLoadBalancer.GetLayersRequest\x1a\".DNNLoadBalancer.GetLayersResponse\x12^\n\rGetPrediction\x12%.DNNLoadBalancer.GetPredictionRequest\x1a&.DNNLoadBalancer.GetPredictionResponse\x12U\n\nGetResults\x12\".DNNLoadBalancer.GetResultsRequest\x1a#.DNNLoadBalancer.GetResultsResponseb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'loadbalance_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _QUERY._serialized_start=38
  _QUERY._serialized_end=162
  _RESULT._serialized_start=164
  _RESULT._serialized_end=224
  _LAYER._serialized_start=226
  _LAYER._serialized_end=260
  _LAYERS._serialized_start=262
  _LAYERS._serialized_end=310
  _GETLAYERSREQUEST._serialized_start=312
  _GETLAYERSREQUEST._serialized_end=385
  _GETLAYERSRESPONSE._serialized_start=387
  _GETLAYERSRESPONSE._serialized_end=478
  _GETPREDICTIONREQUEST._serialized_start=480
  _GETPREDICTIONREQUEST._serialized_end=604
  _GETPREDICTIONRESPONSE._serialized_start=606
  _GETPREDICTIONRESPONSE._serialized_end=696
  _GETRESULTSREQUEST._serialized_start=698
  _GETRESULTSREQUEST._serialized_end=796
  _LISTOFFLOATS._serialized_start=798
  _LISTOFFLOATS._serialized_end=828
  _GETRESULTSRESPONSE._serialized_start=831
  _GETRESULTSRESPONSE._serialized_end=1024
  _DNNLOADBALANCER._serialized_start=1027
  _DNNLOADBALANCER._serialized_end=1375
# @@protoc_insertion_point(module_scope)
