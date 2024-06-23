open Torch

type layer_norm = {
  gamma : nn.Parameter.t;
  beta : Tensor.t;
}

let create_layer_norm (dim : int) : layer_norm =
  {
    gamma = nn.Parameter.create (Tensor.ones [dim]);
    beta = Tensor.zeros [dim] |> nn.register_buffer "beta";
  }

let forward_layer_norm (x : Tensor.t) (ln : layer_norm) : Tensor.t =
  let shape = Tensor.shape x in
  let gamma = ln.gamma |> Tensor.unsqueeze (List.length shape - 1) in
  let beta = ln.beta |> Tensor.unsqueeze (List.length shape - 1) in
  F.layer_norm x [shape |> List.nth 1] ~gamma ~beta

type multi_headed_rmsnorm = {
  scale : float;
  gamma : nn.Parameter.t;
}

let create_multi_headed_rmsnorm ?(heads : int = 1) (dim : int) : multi_headed_rmsnorm =
  {
    scale = float_of_int dim ** 0.5;
    gamma = nn.Parameter.create (Tensor.ones [heads; 1; dim]);
  }

let forward_multi_headed_rmsnorm (x : Tensor.t) (rmsnorm : multi_headed_rmsnorm) : Tensor.t =
  let normalized = F.normalize x ~dim:(-1) in
  Tensor.(normalized *$ rmsnorm.scale *$ rmsnorm.gamma)

type learned_sinusoidal_pos_emb = {
  weights : nn.Parameter.t;
}

let create_learned_sinusoidal_pos_emb (dim : int) : learned_sinusoidal_pos_emb =
  assert ((dim mod 2) = 0);
  let half_dim = dim / 2 in
  {
    weights = nn.Parameter.create (Tensor.randn [half_dim]);
  }

let forward_learned_sinusoidal_pos_emb (x : Tensor.t) (pos_emb : learned_sinusoidal_pos_emb) : Tensor.t =
  let x = Rearrange.forward x "b -> b 1" in
  let freqs = Tensor.(x *$ Rearrange.forward pos_emb.weights "d -> 1 d" *$ (2. *$ Float.pi)) in
  let sin_part = Tensor.sin freqs in
  let cos_part = Tensor.cos freqs in
  let fouriered = Tensor.cat [x; sin_part; cos_part] ~dim:(-1) in
  fouriered