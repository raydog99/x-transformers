open Torch

let exists v = Torch.(v <> none)

let default v d = if exists v then v else d

let feedForward
    ?(dim : int option)
    ?(mult : int = 4)
    ?(use_coor_descent : bool = false)
    ?(coor_descent_iters : int = 20)
    ?(coor_descent_sparsity_k : int option = None)
    ?(coor_descent_eps : float = 1e-1)
    ?(coor_descent_eps_init : float = 4.)
    ?(coor_descent_eps_decay : float = 0.7)
    ()
  =
  let dim_hidden = match dim with
    | Some d -> d * mult
    | None -> failwith "Dimension 'dim' must be provided."
  in

  let coor_descent_sparsity_k = match coor_descent_sparsity_k with
    | Some k -> k
    | None -> dim_hidden / 10
  in

  let proj_in =
    sequential
      [
        LayerNorm LayerNormConfig.(make_layer_norm ~features:dim ());
        linear ~input_dim:dim ~output_dim:dim_hidden (LinearConfig.create ());
      ]
  in

  let proj_out = linear ~input_dim:dim_hidden ~output_dim:dim (LinearConfig.create ()) in

  let forward x =
    let x = x |> forward proj_in |> Tensor.to_tensor |> Tensor.get 0 |> Tensor.to_tensor in
    if use_coor_descent then
      let x =
        triton_coor_descent x
          ~n_iters:coor_descent_iters
          ~k:coor_descent_sparsity_k
          ~eps:coor_descent_eps
          ~eps_init:coor_descent_eps_init
          ~eps_decay:coor_descent_eps_decay
          ~checkpoint_segments:(coor_descent_iters / 5)
      in
      x
    else x |> F.gelu |> forward proj_out
  in
  forward