open Base
open Torch

module NesT = struct
  type t = {
    to_patch_embedding : nn;
    layers : (Transformer.t * nn) list;
    mlp_head : nn;
  }

  let create
      ~image_size
      ~patch_size
      ~num_classes
      ~dim
      ~heads
      ~num_hierarchies
      ~block_repeats
      ?(mlp_mult=4)
      ?(channels=3)
      ?(dim_head=64)
      ?(dropout=0.0)
      () =
    assert (image_size mod patch_size = 0);
    let num_patches = (image_size / patch_size) * (image_size / patch_size) in
    let patch_dim = channels * patch_size * patch_size in

    let to_patch_embedding =
      nn_sequential [
        Rearrange ("b c (h p1) (w p2) -> b (p1 p2 c) h w", ["p1", patch_size; "p2", patch_size]);
        LayerNorm patch_dim;
        nn_conv2d patch_dim layer_dims.(0) 1;
        LayerNorm layer_dims.(0);
      ]
    { to_patch_embedding }
end