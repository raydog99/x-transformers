open Base
open Torch

module type MAX_VIT = sig
  type t

  val create
    :  num_classes:int
    -> dim:int
    -> depth:int list
    -> dim_head:int
    -> dim_conv_stem:int option
    -> window_size:int
    -> mbconv_expansion_rate:float
    -> mbconv_shrinkage_rate:float
    -> dropout:float
    -> channels:int
    -> t

  val forward : t -> Tensor.t -> Tensor.t
end

module MaxViT : MAX_VIT = struct
  type t = {
    conv_stem : nn;
    layers : nn list;
    mlp_head : nn;
  }

  val create
    :  num_classes:int
    -> dim:int
    -> depth:int list
    -> dim_head:int
    -> dim_conv_stem:int option
    -> window_size:int
    -> mbconv_expansion_rate:float
    -> mbconv_shrinkage_rate:float
    -> dropout:float
    -> channels:int
    -> t
end