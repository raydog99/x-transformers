open Base
open Torch

let separate_weight_decayable_params (params : Tensor.t list) : Tensor.t list * Tensor.t list =
  List.fold params ~init:([], []) ~f:(fun (wd_params, no_wd_params) param ->
    if Tensor.ndim param < 2 then
      (no_wd_params @ [param], wd_params)
    else
      (wd_params @ [param], no_wd_params)
  )

let get_optimizer
    ?(lr : float = 1e-4)
    ?(wd : float = 1e-2)
    ?(betas : (float * float) = (0.9, 0.99))
    ?(eps : float = 1e-8)
    ?(filter_by_requires_grad : bool = false)
    ?(group_wd_params : bool = true)
    ?(kwargs : (string * 'a) list = [])
    (params : Tensor.t list)
  : Optimizer.t =
  let has_weight_decay = Float.( > ) wd 0.0 in

  let params =
    if filter_by_requires_grad then
      List.filter params ~f:(fun t -> Tensor.requires_grad t)
    else
      params
  in

  let adam_kwargs = Adam.{lr = lr; betas = betas; eps = eps; weight_decay = if has_weight_decay then wd else 0.0} in

  if group_wd_params && has_weight_decay then begin
    let wd_params, no_wd_params = separate_weight_decayable_params params in
    let wd_params = List.map wd_params ~f:(fun t -> Adam.Param t) in
    let no_wd_params = List.map no_wd_params ~f:(fun t -> Adam.No_decay t) in
    Optimizer.adam
      (List.concat [wd_params; no_wd_params])
      ~lr:adam_kwargs.lr
      ~betas:adam_kwargs.betas
      ~eps:adam_kwargs.eps
      ~weight_decay:wd
      ~kwargs:kwargs
  end else if has_weight_decay then
    Optimizer.adamw
      params
      ~lr:adam_kwargs.lr
      ~betas:adam_kwargs.betas
      ~eps:adam_kwargs.eps
      ~weight_decay:wd
      ~kwargs:kwargs
  else
    Optimizer.adam
      params
      ~lr:adam_kwargs.lr
      ~betas:adam_kwargs.betas
      ~eps:adam_kwargs.eps
      ~weight_decay:0.0
      ~kwargs:kwargs