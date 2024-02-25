open Torch

let () =
	let vs = Var_store.create () in
	let config = Config.default_config in
	let transformer_model = Transformer.create ~vs ~config in

	let input_ids = Tensor.randint_low 
		~low:0 
		~high:config.Config.vocab_size [ 2; 3; 512 ] 
		~options:(T Int64, Device.Cpu) in
	let layer_past = None in
	let attention_mask = None in
	let token_type_ids = None in
	let position_ids = None in
	let is_training = true in

	let output, _ = transformer_model input_ids 
		~layer_past ~attention_mask ~token_type_ids ~position_ids ~is_training in
	Printf.printf "Input Shape: %s\n" (Tensor.shape_str input_ids);
	Printf.printf "Output Shape: %s\n" (Tensor.shape_str output)
;;