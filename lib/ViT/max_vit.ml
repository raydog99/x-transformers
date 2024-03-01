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

    let num_stages = List.length depth in
    let dims =
      List.init num_stages ~f:(fun i -> Int.( * ) (1 lsl i) dim)
      |> List.cons dim_conv_stem
    in
    let dim_pairs = List.zip_exn dims (List.tl_exn dims) in

    let layers =
      List.concat_mapi depth ~f:(fun ind (layer_dim_in, layer_dim) ->
        List.init layer_dim ~f:(fun stage_ind ->
          let is_first = stage_ind = 0 in
          let stage_dim_in = if is_first then layer_dim_in else layer_dim in

          let block =
            nn_sequential [
              MBConv.create
                ~dim_in:stage_dim_in
                ~dim_out:layer_dim
                ~downsample:is_first
                ~expansion_rate:mbconv_expansion_rate
                ~shrinkage_rate:mbconv_shrinkage_rate
                ~dropout;
              Rearrange.create "b d (x w1) (y w2) -> b x y w1 w2 d" ~w1:window_size ~w2:window_size;
              Residual.create (Attention.create ~dim:layer_dim ~dim_head ~dropout ~window_size);
              Residual.create (FeedForward.create ~dim:layer_dim ~dropout);
              Rearrange.create "b x y w1 w2 d -> b d (x w1) (y w2)";
              Rearrange.create "b d (w1 x) (w2 y) -> b x y w1 w2 d" ~w1:window_size ~w2:window_size;
              Residual.create (Attention.create ~dim:layer_dim ~dim_head ~dropout ~window_size);
              Residual.create (FeedForward.create ~dim:layer_dim ~dropout);
              Rearrange.create "b x y w1 w2 d -> b d (w1 x) (w2 y)";
            ]
          in
          block
        )
      )
    in
    { conv_stem; layers }
  ;;
end