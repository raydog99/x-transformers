open Torch

let classifier_free_guidance
    (fn: ('a -> 'b) Layer.t)
    ?(cond_drop_prob_keyname="cond_drop_prob")
    ?(texts_key_name="texts")
    ?(text_embeds_key_name="text_embeds")
    ?(cond_fns_keyname="cond_fns")
    ?(text_conditioner_name="text_conditioner")
    () : ('a -> 'b) Layer.t =
  let fn_params = Layer.signature fn in
  let auto_handle_text_condition =
    not (List.mem texts_key_name fn_params) &&
    not (List.mem text_embeds_key_name fn_params)
  in
  let inner (self: Layer.t) ~(cond_scale: float) ?(rescale_phi=0.0)
      ?(return_unconditioned=false) ?(cfg_routed_kwargs=[]) (x: 'a) : 'b =
    let fn_maybe_with_text (self: Layer.t) (x: 'a) : 'b =
      if auto_handle_text_condition then
        let texts = Layer.get_attr texts_key_name self in
        let text_embeds = Layer.get_attr text_embeds_key_name self in
        assert (Option.is_none texts || Option.is_none text_embeds);
        let cond_fns, raw_text_cond =
          match Layer.get_attr_opt text_conditioner_name self with
          | Some text_conditioner ->
            let cond_drop_prob = Layer.get_attr_opt cond_drop_prob_keyname self in
            assert (Option.is_none cond_drop_prob || cond_drop_prob >= 0.0 && cond_drop_prob <= 1.0);
            if Option.is_some texts || Option.is_some text_embeds then
              let text_condition_input =
                if Option.is_some texts then [texts_key_name, Layer.get_attr texts_key_name self]
                else [text_embeds_key_name, Layer.get_attr text_embeds_key_name self]
              in
              Layer.forward text_conditioner text_condition_input
            else
              Layer.forward text_conditioner []
          | None -> failwith "text_conditioner must be set"
        in
        Layer.set_attr cond_fns_keyname cond_fns self;
        Layer.set_attr "raw_text_cond" raw_text_cond self;
        Layer.forward fn x
      else
        Layer.forward fn x
    in
    if Layer.is_training self then
      assert (cond_scale = 1.0);
      fn_maybe_with_text self x
    else
      assert (cond_scale >= 1.0);
      let kwargs_without_cond_dropout = List.map (fun (k, v) -> if k = cond_drop_prob_keyname then (k, 0.0) else (k, v)) cfg_routed_kwargs in
      let kwargs_with_cond_dropout = List.map (fun (k, v) -> if k = cond_drop_prob_keyname then (k, 1.0) else (k, v)) cfg_routed_kwargs in
      let outputs = fn_maybe_with_text self kwargs_without_cond_dropout x in
      if cond_scale = 1.0 then
        outputs
      else
        let logits, rest = Layer.cast_tuple outputs in
        let null_outputs = fn_maybe_with_text self kwargs_with_cond_dropout x in
        let null_logits, null_rest = Layer.cast_tuple null_outputs in
        let scaled_logits = Layer.add (Layer.sub logits null_logits) (Layer.mul null_logits cond_scale) in
        let logit_output =
          if rescale_phi <= 0.0 then
            scaled_logits
          else
            let dims = Array.init (Layer.rank logits - 2) (fun i -> i + 1) in
            let rescaled_logits = Layer.mul scaled_logits (Layer.div (Layer.std ~dims logits) (Layer.std ~dims scaled_logits)) in
            Layer.add (Layer.mul rescaled_logits rescale_phi) (Layer.mul scaled_logits (1.0 -. rescale_phi))
        in
        let output = if return_unconditioned then (logit_output, null_logits) else logit_output in
        if List.length rest = 0 then
          output
        else
          (output :: rest)
  in
  Layer.wrap (inner self)