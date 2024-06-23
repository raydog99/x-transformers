open Torch
open Tensor
open Layer

let loss_fn x y =
  let x = Tensor.(normalize x ~p:2. ~dim:(-1) ~eps:1e-12) in
  let y = Tensor.(normalize y ~p:2. ~dim:(-1) ~eps:1e-12) in
  let xy = Tensor.(sum (x * y) ~dim:[-1] ~keepdim:false) in
  Tensor.(mul_scalar (sub_scalar xy 2.) (-2.))

module RandomApply = struct
  type t = {
    fn: Tensor.t -> Tensor.t;
    p: float;
  }

  let create ~fn ~p = { fn; p }

  let forward t x =
    if Random.float 1.0 > t.p then x else t.fn x
end

module EMA = struct
  type t = {
    beta: float;
  }

  let create ~beta = { beta }

  let update_average t ~old ~new_ =
    match old with
    | None -> new_
    | Some old -> Tensor.(add (mul_scalar old t.beta) (mul_scalar new_ (1. -. t.beta)))
end

let update_moving_average ema_updater ma_model current_model =
  List.iter2 (fun current_params ma_params ->
    let old_weight = Tensor.data ma_params in
    let up_weight = Tensor.data current_params in
    Tensor.copy_ ma_params (EMA.update_average ema_updater ~old:(Some old_weight) ~new_:up_weight)
  ) (Module.parameters current_model) (Module.parameters ma_model)

module MaybeSyncBatchnorm = struct
  type t = {
    sync_batchnorm: bool option;
  }

  let create sync_batchnorm = { sync_batchnorm }

  let forward t x =
    match t.sync_batchnorm with
    | Some true -> Layer.batch_norm1d x
    | _ -> x
end

let mlp ~dim ~projection_size ?(hidden_size=4096) ?sync_batchnorm () =
  let open Layer in
  Sequential.sequential [
    Linear.linear dim hidden_size;
    MaybeSyncBatchnorm.create sync_batchnorm;
    Relu.relu ();
    Linear.linear hidden_size projection_size
  ]

let sim_siam_mlp ~dim ~projection_size ?(hidden_size=4096) ?sync_batchnorm () =
  let open Layer in
  Sequential.sequential [
    Linear.linear ~bias:false dim hidden_size;
    MaybeSyncBatchnorm.create sync_batchnorm;
    Relu.relu ();
    Linear.linear ~bias:false hidden_size hidden_size;
    MaybeSyncBatchnorm.create sync_batchnorm;
    Relu.relu ();
    Linear.linear ~bias:false hidden_size projection_size;
    MaybeSyncBatchnorm.create ~affine:false sync_batchnorm
  ]