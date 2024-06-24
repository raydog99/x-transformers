open Torch

module SlicedScoreMatching = struct
  let grad f x =
    let x = Tensor.set_requires_grad x true in
    let y = f x in
    Tensor.backward y;
    Tensor.grad x

  let sliced_score_matching p_m x v =
    let s_m = grad (fun x -> Tensor.log (p_m x)) x in
    let v_t_nabla_x_s_m = grad (fun x -> Tensor.dot v (s_m x)) x in
    let j1 = Tensor.mul_scalar (Tensor.dot v s_m) (Scalar.float 0.5) in
    let j2 = Tensor.dot v v_t_nabla_x_s_m in
    Tensor.add j1 j2

  let run p_m x v =
    let j = sliced_score_matching p_m x v in
    Tensor.to_float0_exn j
end