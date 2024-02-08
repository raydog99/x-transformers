module TrainingConfig = struct
	type t = {
		vocab_size: int;
		device : Device.t;
	}

	let default_config = {
		vocab_size = 10000;
		device : Device.cuda_if_available()
	}
end