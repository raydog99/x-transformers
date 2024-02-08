open Torch

module Encoder = struct
	type t = {
		num_layers = Config.num_layers;
		vocab_size = trainingConfig.Config.vocab_size;
		num_embeddings = modelConfig.Config.num_embeddings;
		num_positions = modelConfig.Config.num_positions;
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

	let create vs config =
		let tokenEmbedding = Layer.embeddings Var_store.(vs / "wte") 
			~num_embeddings: vocab_size
			~embedding_dim: num_embeddings in
		let positionEncoder = Layer.embeddings Var_store.(vs / "wpe")
			~num_embeddings: num_positions
			~embedding_dim: num_embeddings in
		List.init num_layers ~f (fun layer_idx -> 
	      Encoder.create ~vs:(Var_store.vs( / Int.to_string layer_idx)) ~config)
;;
end