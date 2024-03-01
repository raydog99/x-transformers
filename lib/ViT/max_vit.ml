open Base
open Torch

module MaxViT = struct
  type t = {
    conv_stem : nn;
    layers : nn list;
    mlp_head : nn;
  }

  let create
      ~num_classes
      ~dim
      ~depth
      ~dim_head
      ~dim_conv_stem
      ~window_size
      ~mbconv_expansion_rate
      ~mbconv_shrinkage_rate
      ~dropout
      ~channels =

    let conv_stem =
      nn_sequential [
        nn_conv2d ~stride:2 ~padding:1 ~kernel_size:3 channels dim_conv_stem;
        nn_conv2d ~padding:1 ~kernel_size:3 dim_conv_stem dim_conv_stem;
      ]
    { conv_stem }
  ;;
end