module type Optimizer = sig
  type t

  val init : params:t -> lr:float -> betas:(float * float) -> weight_decay:float -> t
  val step : t -> (unit -> float option) -> unit
end

module Lion (Optimizer : Optimizer) = struct
  type t = {
    params : Optimizer.t;
    lr : float;
    betas : float * float;
    weight_decay : float;
    mutable decoupled_wd : bool;
    mutable init_lr : float;
    update_fn : (Optimizer.t -> float -> float -> float -> float -> float -> unit);
  }

  let init ~params ~lr ~betas ~weight_decay ~use_triton ~decoupled_weight_decay =
    assert (lr > 0.);
    assert (0. <= fst betas && fst betas <= 1.);
    assert (0. <= snd betas && snd betas <= 1.);

    let defaults = {
      lr = lr;
      betas = betas;
      weight_decay = weight_decay;
    } in

    let update_fn = fun _p _grad _exp_avg _lr _wd _beta1 _beta2 -> () in

    let update_fn =
      if use_triton then
        (* Import update_fn from triton *)
        update_fn
      else
        update_fn
    in

    {
      params = Optimizer.init ~params ~lr ~betas ~weight_decay;
      lr;
      betas;
      weight_decay;
      decoupled_wd = decoupled_weight_decay;
      init_lr = lr;
      update_fn;
    }

  let step self closure =
    let loss =
      match closure with
      | Some closure ->
          (* Evaluate closure *)
          let loss_value = closure () in
          Some loss_value
      | None -> None
    in

    List.iter (fun group ->
        List.iter (fun p ->
            (* Perform optimizer step for each parameter *)
            let grad = (* Compute gradient *) in
            let lr = group.lr in
            let wd = group.weight_decay in
            let beta1 = fst group.betas in
            let beta2 = snd group.betas in
            let state = (* Get state for parameter *) in
            let decoupled_wd = self.decoupled_wd in
            let init_lr = self.init_lr in

            (* Maybe apply decoupled weight decay *)
            if decoupled_wd then
              wd /. init_lr;

            (* Initialize state if necessary *)
            (* E.g., exp_avg initialization *)

            (* Call update function *)
            self.update_fn
              p
              grad
              exp_avg
              lr
              wd
              beta1
              beta2
          ) group.params
      ) self.params;

    loss
end