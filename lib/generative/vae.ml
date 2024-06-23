(* Standard library modules *)
open Random;;
open Stats;;

(* Data type for variational parameters *)
type var_param = {
  mu: float array; (* Mean vector *)
  sigma: float array; (* Standard deviation vector *)
};

(* Function to initialize variational parameters *)
let init_var_param (dim: int) : var_param = {
  mu = Array.init dim (fun _ -> Random.float);;
  sigma = Array.init dim (fun _ -> 1.0);;
};

(* Function to compute the evidence lower bound (ELBO) *)
let elbo (data: float array array; param: var_param) : float =
  let (d, n) = (Array.length data.(0), Array.length data) in
  let (mu, sigma) = (param.mu, param.sigma) in
  let kl_div = 
    Array.fold_left (fun acc x -> acc +. 0.5 * (x *. log x -. 1.0)) 0.0 sigma in (* KL divergence term *)
  let recon_term = 
    Array.fold_left (fun acc i -> 
      let diff = data.(i) -. mu in
      acc +. Array.fold_left (fun acc j -> acc +. diff.(j) *. diff.(j) /. (2.0 *. sigma.(j) *. sigma.(j)) 0.0 d
    ) 0.0 n in
  (* Combine terms and negate for minimization *)
  - (kl_div +. recon_term);

(* Function to perform a single SGVB update step *)
let sgvb_update (data: float array; param: var_param) : var_param =
  let (d, _) = (Array.length data, Array.length param.mu) in
  let (mu, sigma) = (param.mu, param.sigma) in
  let new_mu = Array.init d (fun i -> 
    mu.(i) + sigma.(i) *. (Array.fold_left (fun acc x -> acc +. x.(i) / n in
                                             0.0 data)  - mu.(i)) in
  let new_sigma = Array.init d (fun i -> 
    sigma.(i) * sqrt (1.0 + (data.(.) .(i) -. mu.(i)) **2.0 /. (sigma.(i) *. sigma.(i)) ) in
  {mu = new_mu; sigma = new_sigma};

(* Function to perform AEVG inference *)
let aevb (data: float array array; num_iters: int) : var_param =
  let init_param = init_var_param (Array.length data.(0)) in
  let rec loop (param: var_param; iter: int) : var_param =
    if iter >= num_iters then
      param
    else
      let updated_param = sgvb_update (data.(Random.int n), param) in
      loop (updated_param, iter + 1) in
  loop (init_param, 0);