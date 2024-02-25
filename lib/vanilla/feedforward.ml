open Torch

module Feedforward = struct
	let { Config.resid_p; n_embd; _ } = config in
	let activation = Activation.gelu_new in

	Layer.of_fn_ (fun xs ~is_training ->
      Layer.forward c_fc xs
      |> activation
      |> Layer.forward c_proj
      |> Tensor.dropout ~p:resid_p ~is_training)
end