open Base

module type Module = sig
  type t
  val forward : t -> float array array -> float array array
end

module RMSNorm : sig
  type t
  val create : int -> t
  val forward : t -> float array array -> float array array
end = struct
  type t = {
    scale : float;
    gamma : float array;
  }

  let create dim =
    let scale = sqrt (float_of_int dim) in
    let gamma = Array.make dim 1.0 in
    { scale; gamma }

  let l2_norm x =
    let sum = Array.fold_left (fun acc xi -> acc +. xi *. xi) 0.0 x in
    sqrt sum

  let normalize x =
    Array.map (fun xi -> xi /. (l2_norm x +. 1e-10)) x

  let forward { scale; gamma } x =
    Array.map (fun xi ->
      let norm_xi = normalize xi in
      Array.mapi (fun i xi -> xi *. scale *. gamma.(i)) norm_xi
    ) x
end

module FeedForward : sig
  type t
  val create : int -> ?mult:int -> ?dropout:float -> unit -> t
  val forward : t -> float array array -> float array array
end = struct
  type t = {
    norm : RMSNorm.t;
    proj_in : float array array -> float array array * float array array;
    proj_out : float array array -> float array array;
    dropout : float array array -> float array array;
  }

  let create_linear in_dim out_dim x =
    Array.init (Array.length x) (fun _ ->
      Array.init out_dim (fun _ -> Random.float 1.0)
    )

  let gelu x =
    let gelu_single xi =
      0.5 *. xi *. (1.0 +. tanh (sqrt (2.0 /. Float.pi) *. (xi +. 0.044715 *. xi ** 3.0)))
    in
    Array.map gelu_single x

  let create dim ?(mult=4) ?(dropout=0.0) () =
    let dim_inner = int_of_float (float_of_int (mult * dim) *. 2.0 /. 3.0) in
    let norm = RMSNorm.create dim in
    let proj_in x =
      let linear = create_linear dim (dim_inner * 2) x in
      let split_at = dim_inner in
      let x1 = Array.map (fun xi -> Array.sub xi 0 split_at) linear in
      let x2 = Array.map (fun xi -> Array.sub xi split_at split_at) linear in
      x1, x2
    in
    let proj_out x =
      create_linear dim_inner dim x
    in
    let dropout x =
      Array.map (fun xi ->
        Array.map (fun xij ->
          if Random.float 1.0 < dropout then 0.0 else xij
        ) xi
      ) x
    in
    { norm; proj_in; proj_out; dropout }

  let forward { norm; proj_in; proj_out; dropout } x =
    let x = RMSNorm.forward norm x in
    let x, gates = proj_in x in
    let x = Array.map2 (fun xi gi -> Array.map2 (fun xij gij -> xij *. gelu gij) xi gi) x gates in
    let x = dropout x in
    proj_out x
end

let retrieve_from_kv_memories t (past_memories_kv, past_memories_norm) eps =
  let einsum_numerator t kv =
    Array.mapi (fun i ti ->
      Array.init (Array.length kv.(0).(0)) (fun n ->
        Array.fold_left (+.) 0.0 (Array.mapi (fun j tij ->
          Array.fold_left (+.) 0.0 (Array.mapi (fun k tik -> tik *. kv.(i).(k).(n)) tij)
        ) ti)
      )
    ) t
  in

  let einsum_denominator t norm =
    Array.mapi (fun i ti ->
      Array.map (fun tij ->
        Array.fold_left (+.) 0.0 (Array.mapi (fun k tik -> tik *. norm.(i).(k)) tij)
      ) ti
    ) t
  in

  let numer = einsum_numerator t past_memories_kv in
  let denom = einsum_denominator t past_memories_norm in
  let denom = Array.map (fun di -> Array.map (fun d -> max d eps) di) denom in
  Array.mapi (fun i ni -> Array.map2 (/. ) ni denom.(i)) numer