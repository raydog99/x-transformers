open Torch

module type MODULE = sig
  type t
  val forward : t -> Tensor.t -> Tensor.t
end

module LinearNoBiasThenOuterSum = struct
  type t = { proj : Linear.t }

  let create dim dim_out =
    let dim_out = Option.value ~default:dim dim_out in
    { proj = Linear.create ~bias:false dim (dim_out * 2) }

  let forward { proj } t =
    let single_i, single_j = Tensor.chunk (Linear.forward proj t) 2 ~dim:(-1) in
    Tensor.(single_i + single_j)
end

module SwiGLU = struct
  type t = unit

  let create () = ()

  let forward () x =
    let x, gates = Tensor.chunk x 2 ~dim:(-1) in
    Tensor.(silu gates * x)
end

module Transition = struct
  type t = { ff : Layer.t }

  let create ~dim ~expansion_factor =
    let dim_inner = dim * expansion_factor in
    let ff = Layer.of_list [
      Layer.linear_no_bias ~input_dim:dim (dim_inner * 2);
      Layer.fn (fun x -> SwiGLU.forward () x);
      Layer.linear_no_bias ~input_dim:dim_inner dim;
    ] in
    { ff }

  let forward { ff } x = Layer.forward ff x
end

module Dropout = struct
  type t = { dropout : Dropout.t; dropout_type : string option }

  let create ~prob ~dropout_type =
    let dropout = Dropout.create ~p:prob in
    { dropout; dropout_type }

  let forward { dropout; dropout_type } t =
    match dropout_type with
    | Some "row" ->
      let batch, row, _, _ = Tensor.shape4_exn t in
      let ones_shape = [batch; row; 1; 1] in
      let ones = Tensor.ones ones_shape ~device:(Tensor.device t) in
      let dropped = Dropout.forward dropout ones in
      Tensor.(t * dropped)
    | Some "col" ->
      let batch, _, col, _ = Tensor.shape4_exn t in
      let ones_shape = [batch; 1; col; 1] in
      let ones = Tensor.ones ones_shape ~device:(Tensor.device t) in
      let dropped = Dropout.forward dropout ones in
      Tensor.(t * dropped)
    | _ -> Dropout.forward dropout t
end

module PreLayerNorm = struct
  module type LAYER = sig
    type t
    val forward : t -> Tensor.t -> Tensor.t
  end

  type 'a t = {
    norm : Layer_norm.t;
    fn : 'a
  }

  let create (type a) (module Layer : LAYER with type t = a) fn ~dim =
    let norm = Layer_norm.create dim in
    { norm; fn }

  let forward { norm; fn } x =
    let x = Layer_norm.forward norm x in
    fn x
end

module AdaptiveLayerNorm = struct
  type t = {
    norm : Layer_norm.t;
    norm_cond : Layer_norm.t;
    to_gamma : Layer.t;
    to_beta : Linear.t
  }

  let create ~dim ~dim_cond =
    let norm = Layer_norm.create ~elementwise_affine:false dim in
    let norm_cond = Layer_norm.create ~bias:false dim_cond in
    let to_gamma = Layer.of_list [
      Layer.linear ~input_dim:dim_cond dim;
      Layer.sigmoid ();
    ] in
    let to_beta = Linear.create ~bias:false dim_cond dim in
    { norm; norm_cond; to_gamma; to_beta }

  let forward { norm; norm_cond; to_gamma; to_beta } x cond =
    let normed = Layer_norm.forward norm x in
    let normed_cond = Layer_norm.forward norm_cond cond in
    let gamma = Layer.forward to_gamma normed_cond in
    let beta = Linear.forward to_beta normed_cond in
    Tensor.(normed * gamma + beta)
end