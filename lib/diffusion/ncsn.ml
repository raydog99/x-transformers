open Torch

let annealed_langevin_dynamics sigmas epsilon t s_theta =
  let l = Array.length sigmas in
  let x_0 = Tensor.zeros [1] in
  
  let rec outer_loop i x_0 =
    if i >= l then x_0
    else
      let alpha_i = epsilon *. (sigmas.(i) ** 2.) /. (sigmas.(l-1) ** 2.) in
      
      let rec inner_loop t x_t =
        if t > t then x_t
        else
          let z_t = Tensor.randn [1] in
          let s_theta_term = Tensor.mul_scalar (s_theta x_t sigmas.(i)) (Scalar.float (alpha_i /. 2.)) in
          let noise_term = Tensor.mul_scalar z_t (Scalar.float (sqrt alpha_i)) in
          let x_t_next = Tensor.(x_t + s_theta_term + noise_term) in
          inner_loop (t + 1) x_t_next
      in
      
      let x_t = inner_loop 1 x_0 in
      outer_loop (i + 1) x_t
  in
  
  outer_loop 0 x_0

let s_theta x sigma =
  Tensor.neg x