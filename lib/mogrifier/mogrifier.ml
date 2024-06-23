module type Module = sig
  type t

  val forward : t -> Tensor.t -> Tensor.t -> int option -> Tensor.t * Tensor.t
end

module Mogrifier (Module : Module) = struct
  type t = {
    dim : int;
    dim_hidden : int;
    iters : int;
    q : Module.t;
    r : Module.t;
  }

  let init ~dim ~iters ?(factorize_k : int option) ?(dim_hidden : int option) ?(hidden_factorize_k : int option) () =
    assert (iters > 1);

    let dim_hidden = match dim_hidden with
      | Some dh -> dh
      | None -> dim
    in

    let q = Module.initialize_q in (* Initialize Q module *)
    let r = Module.initialize_r in (* Initialize R module *)

    {
      dim;
      dim_hidden;
      iters;
      q;
      r;
    }

  let forward self inputs hiddens iters =
    let iters = match iters with
      | Some i -> i
      | None -> self.iters
    in

    assert (Tensor.shape inputs |> List.rev |> List.hd = self.dim);
    assert (Tensor.shape hiddens |> List.rev |> List.hd = self.dim_hidden);
    assert (List.rev (Tensor.shape inputs) |> List.tl |> List.rev = List.rev (Tensor.shape hiddens) |> List.tl);

    for ind = 0 to self.iters - 1 do
      let is_even = (ind mod 2) == 0 in

      if is_even then
        inputs = 2 * self.q(hiddens) * inputs
      else
        hiddens = 2 * self.r(inputs) * hiddens
    done;

    (inputs, hiddens)
end