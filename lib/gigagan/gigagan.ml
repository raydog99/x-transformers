open Core

let gradient_penalty
    (images : Tensor.t)
    (outputs : Tensor.t list)
    ?(grad_output_weights : Tensor.t list option)
    ?(weight : float = 10.)
    ?(scaler : GradScaler.t option)
    ?(eps : float = 1e-4)
    () : Tensor.t =
  let outputs =
    match outputs with
    | [] -> failwith "Empty outputs list"
    | _ -> outputs
  in

  let outputs =
    match scaler with
    | Some scaler -> List.map scaler.scale outputs
    | None -> outputs
  in

  let grad_output_weights =
    match grad_output_weights with
    | Some weights -> weights
    | None -> List.init (List.length outputs) (fun _ -> Tensor.ones_like (List.hd outputs) * weight)
  in

  let maybe_scaled_gradients, _ =
    torch_grad
      ~outputs
      ~inputs:images
      ~grad_outputs:(List.map2 (fun out weight -> Tensor.mul (Tensor.ones_like out) weight) outputs grad_output_weights)
      ~create_graph:true
      ~retain_graph:true
      ~only_inputs:true
      ()
  in

  let gradients =
    match scaler with
    | Some scaler ->
        let scale = GradScaler.get_scale scaler in
        let inv_scale = 1. /. max scale eps in
        Tensor.mul maybe_scaled_gradients (Tensor.of_float inv_scale)
    | None -> maybe_scaled_gradients
  in

  Tensor.rearrange gradients ~pattern:"b ... -> b (...)"

let generator_hinge_loss (fake : Tensor.t) : Tensor.t =
  Tensor.mean fake

let discriminator_hinge_loss (real : Tensor.t) (fake : Tensor.t) : Tensor.t =
  let open Tensor in
  let one = ones_like real in
  let minus_one = ones_like real * -1. in
  let zero = zeros_like real in
  mean (relu (add real one) + relu (add fake minus_one))