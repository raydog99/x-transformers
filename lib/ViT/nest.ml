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
    in
    let block_repeats = cast_tuple block_repeats num_hierarchies in
    let layers =
      List.map3_exn
        hierarchies
        layer_heads
        dim_pairs
        block_repeats
        ~f:(fun level heads (dim_in, dim_out) block_repeat ->
          let is_last = level = 0 in
          let depth = block_repeat in
          let transformer = Transformer.create ~dim_in ~seq_len ~depth ~heads ~mlp_mult ~dropout () in
          let aggregate = if not is_last then Aggregate.create ~dim:dim_in ~dim_out () else nn_identity () in
          (transformer, aggregate))
    in
    { to_patch_embedding; layers }
end