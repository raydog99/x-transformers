open Torch

module Decoder = struct
	type t = {
		num_layers = Config.num_layers;
	}

	let layer vs config ~scale =
		let { Config.num_embeddings; layer_norm_epsilon; _ } = config in
		let ln1 = Layer.layer_norm Var_store.(vs / "ln_1") num_embeddings ~eps:layer_norm_epsilon in
		let ln2 = Layer.layer_norm Var_store.(vs / "ln_2") num_embeddings ~eps:layer_norm_epsilon in
		let attention = MultiheadAttention.create Var_store.(vs / "attn") config ~scale in
		let feedforward = Feedforward.create Var_store.(vs / "attn") config ~scale in

		fun xs ~layer_past ~attention_mask ~is_training ->
			let output, present = Layer.forward ln1 xs 
				|> attention ~layer_past ~attention_mask ~is_training in
			let xs = Tensor.(xs + output) in
			let m = Layer.forward ln2 xs |> Layer.forward_ feedforward ~is_training in
			Tensor.(xs + m), present
		;;
end