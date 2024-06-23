open Base
open Torch

let posemb_sincos_1d seq dim ~temperature ~(device : Device.t option) ~(dtype : Kind.packed) =
  let device = Option.value device ~default:Device.Cpu in
  let n = Tensor.arange' 0 `L seq ~device in
  let omega = Tensor.arange' 0 `L (dim / 2) ~device / (Tensor.of_int (dim / 2 - 1) ~device |> Tensor.to_type Float) in
  let omega = Tensor.pow_scalar (Tensor.of_float temperature ~device) omega in

  let n = Tensor.unsqueeze n ~dim:1 in
  let n = Tensor.mul n omega in

  let sin_part = Tensor.sin n in
  let cos_part = Tensor.cos n in

  Tensor.cat [sin_part; cos_part] ~dim:1
  |> Tensor.to_type dtype

let gelu x = Tensor.gelu x
let dropout x p = Tensor.dropout x ~p ~train:true
let linear x weight bias = Tensor.linear x weight bias

let exists x = Option.is_some x

let residual (fn : Tensor.t -> Tensor.t) x =
  Tensor.(fn x + x)

type layer_norm = { gamma : Tensor.t; beta : Tensor.t }

let layer_norm dim =
  let gamma = Tensor.ones [dim] ~device:(Device.Cpu 0) |> Tensor.to_requires_grad ~requires_grad:true in
  let beta = Tensor.zeros [dim] ~device:(Device.Cpu 0) in
  { gamma; beta }

let layer_norm_forward x { gamma; beta } =
  Tensor.layer_norm x ~normalized_shape:(List.init (Tensor.ndim x) ~f:(fun _ -> dim)) ~weight:gamma ~bias:beta ~eps:1e-5

type feed_forward = { norm : layer_norm; net : (Tensor.t -> Tensor.t) list }

let feed_forward dim ~mult ~dropout =
  let inner_dim = dim * mult in
  let norm = layer_norm dim in
  let net =
    [
      linear ~dim ~dim:inner_dim ~dropout;
      gelu;
      dropout ~p:dropout;
      linear ~dim:inner_dim ~dim;
      dropout ~p:dropout;
    ]
  in
  { norm; net }

let feed_forward_forward { norm; net } x cond_fn =
  let x = layer_norm_forward x norm in
  match exists cond_fn with
  | true -> List.fold net ~init:x ~f:(fun acc fn -> fn acc cond_fn)
  | false -> List.fold net ~init:x ~f:(fun acc fn -> fn acc)