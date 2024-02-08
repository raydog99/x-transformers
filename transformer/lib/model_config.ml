module ModelConfig = struct
	type t = {
		num_layers : int;
		num_embeddings : int;
		num_heads : int;
		dim_feedforward : int;
		layer_norm_epsilon : float;
		residual_dropout : float;
		embedding_dropout : float;
		attention_dropout : float;
	}

	let default_config = {
		num_layers = 6;
		num_embeddings = 512;
		num_heads = 8;
		dim_feedforward = 2048;
		layer_norm_epsilon = 1e-5;
		residual_dropout = 0.1;
		embedding_dropout = 0.1;
		attention_dropout = 0.1;
	}
end