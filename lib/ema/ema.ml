open Torch

module EMA = struct
  type t = {
    beta: float;
    mutable is_frozen: bool;
    mutable online_model: Layer.t option;
    ema_model: Layer.t;
    parameter_names: string list;
    buffer_names: string list;
    inplace_copy: Tensor.t -> Tensor.t -> unit;
    inplace_lerp: Tensor.t -> Tensor.t -> float -> unit;
  }

  let create
      ~model
      ?ema_model
      ?(beta=0.9999)
      ?(update_after_step=100)
      ?(update_every=10)
      ?(inv_gamma=1.0)
      ?(power=2. /. 3.)
      ?(min_value=0.0)
      ?(param_or_buffer_names_no_ema=[])
      ?(ignore_names=[])
      ?(ignore_startswith_names=[])
      ?(include_online_model=true)
      ?(allow_different_devices=false)
      ?(use_foreach=false)
      () =

    let is_frozen = (beta = 1.) in

    let online_model = if include_online_model then Some model else None in

    let ema_model = match ema_model with
      | Some em -> em
      | None -> Layer.clone model
    in

    (* Detach parameters *)
    List.iter (fun p -> Tensor.detach_ p) (Layer.parameters ema_model);

    let parameter_names = 
      List.filter_map (fun (name, param) ->
        if Tensor.is_floating_point param || Tensor.is_complex param then Some name else None
      ) (Layer.named_parameters ema_model)
    in

    let buffer_names = 
      List.filter_map (fun (name, buffer) ->
        if Tensor.is_floating_point buffer || Tensor.is_complex buffer then Some name else None
      ) (Layer.named_buffers ema_model)
    in

    let inplace_copy = 
      if allow_different_devices then fun src dst -> Tensor.copy_ ~src dst
      else fun src dst -> Tensor.copy_ ~src dst
    in

    let inplace_lerp =
      if allow_different_devices then fun src dst weight -> Tensor.lerp_ ~src dst ~weight
      else fun src dst weight -> Tensor.lerp_ ~src dst ~weight
    in

    { beta; is_frozen; online_model; ema_model; parameter_names; buffer_names; inplace_copy; inplace_lerp }

  let update t ~step =
    if t.is_frozen then () else
    if step < update_after_step then () else
    if step mod update_every <> 0 then () else

    let one_minus_beta = 1.0 -. t.beta in

    (* Update parameters *)
    List.iter (fun name ->
      match Layer.parameter t.online_model name, Layer.parameter t.ema_model name with
      | Some src, Some dst -> 
          t.inplace_lerp src dst one_minus_beta;
          t.inplace_copy src dst
      | _, _ -> ()
    ) t.parameter_names;

    (* Update buffers *)
    List.iter (fun name ->
      match Layer.buffer t.online_model name, Layer.buffer t.ema_model name with
      | Some src, Some dst -> 
          t.inplace_lerp src dst one_minus_beta;
          t.inplace_copy src dst
      | _, _ -> ()
    ) t.buffer_names

  let forward t x =
    Layer.forward t.ema_model x
end