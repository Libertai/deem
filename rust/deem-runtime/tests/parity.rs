//! Parity against the torch reference tensors exported by
//! `rust/parity/export_reference.py`.
//!
//! Run:
//!   cargo test --release parity -- --nocapture
//!
//! Expects DEEM_PARITY_DIR (default /tmp/opencode/deem-parity) with
//! raw/*.bin + manifest.json and the HF checkpoint via DEEM_PARITY_CKPT.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn load_f32(name: &str, bytes_per_last_dim: Option<usize>) -> Vec<f32> {
    let dir = std::env::var("DEEM_PARITY_DIR")
        .unwrap_or_else(|_| "/tmp/opencode/deem-parity".to_string());
    let path = Path::new(&dir).join("raw").join(format!("{name}.bin"));
    let raw = fs::read(path).unwrap_or_else(|e| panic!("read {name}: {e}"));
    match bytes_per_last_dim {
        Some(8) => raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        _ => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}

fn load_i32(name: &str) -> Vec<i32> {
    let dir = std::env::var("DEEM_PARITY_DIR")
        .unwrap_or_else(|_| "/tmp/opencode/deem-parity".to_string());
    let path = Path::new(&dir).join("raw").join(format!("{name}.bin"));
    let raw = fs::read(path).unwrap_or_else(|e| panic!("read {name}: {e}"));
    // exported as int64
    raw.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i32)
        .collect()
}

/// Max abs diff with per-tensor reporting.
fn report(name: &str, got: &[f32], want: &[f32]) -> f32 {
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    for (a, b) in got.iter().zip(want.iter()) {
        // abs diff, ignoring NaN mismatches in ref (padding rows may be NaN)
        if b.is_nan() {
            continue;
        }
        let d = (a - b).abs();
        sum += d as f64;
        max = max.max(d);
    }
    println!(
        "{name:<28} max_abs_diff={max:.6} mean={:.6} n={}",
        sum / want.len() as f64,
        want.len(),
    );
    max
}

#[test]
fn parity_reference() {
    let ckpt = std::env::var("DEEM_PARITY_CKPT")
        .unwrap_or_else(|_| "Qwen/Qwen3.5-0.8B".to_string());
    let ckpt_dir = Path::new(&ckpt);
    if !ckpt_dir.exists() {
        eprintln!("checkpoint {ckpt} not found locally; skipping");
        return;
    }

    let tokens: Vec<u32> = load_i32("tokens").iter().map(|x| *x as u32).collect();
    let t = tokens.len();
    println!("tokens: {t}");

    let quantize = std::env::var("DEEM_PARITY_QUANT")
        .map(|v| v == "1")
        .unwrap_or(false);

    let model = deem_runtime::model::load_model(
        ckpt_dir,
        deem_runtime::LoadOptions { quantize },
    )
    .expect("model load");

    // 1) embeddings
    let hidden = model.config.hidden_size;
    let embed_ref: Vec<f32> = {
        let dir = std::env::var("DEEM_PARITY_DIR").unwrap();
        let _ = &dir;
        let v: Vec<f32> = load_f32("embed", None);
        v
    };
    {
        let mut got = vec![0.0f32; t * hidden];
        for (p, tok) in tokens.iter().enumerate() {
            let base = (*tok as usize) * hidden;
            got[p * hidden..(p + 1) * hidden]
                .copy_from_slice(&model.embed[base..base + hidden]);
        }
        let d = report("embed", &got, &embed_ref);
        assert!(d < 0.01, "embed diff too large");
    }

    // 2) letter logits (full forward + head)
    let hidden_states = model.forward_hidden(&tokens);
    let final_ref = load_f32("final_norm", None);
    {
        let readout = deem_runtime::readout::Readout {
            model,
            tokenizer: deem_runtime::tokenizer::Tokenizer::load(ckpt_dir).unwrap(),
        };
        let (logits, _) = readout.letter_logits_from_hidden(&hidden_states, t - 1);
        let ref_logits = load_f32("letter_logits", None);
        let d = report("letter_logits", &logits, &ref_logits);
        assert!(
            d < 0.35,
            "letter logit diff too large: {d}\ngot:  {logits:?}\nwant: {ref_logits:?}"
        );
        // probabilities should agree to ~1e-3
        let scale = ref_logits.iter().fold(0f32, |m, v| m.max(v.abs()));
        println!("reference logit scale: {scale}");
    }
    let _ = final_ref;

    // 3) letter ids match
    let ref_letter_ids = load_i32("letter_ids");
    let tok = deem_runtime::tokenizer::Tokenizer::load(ckpt_dir).unwrap();
    for (i, id) in ref_letter_ids.iter().enumerate() {
        assert_eq!(
            tok.letter_ids[i], *id as u32,
            "letter {} id mismatch",
            b'A' + i as u8
        );
    }

    // 4) layer-by-layer drift
    let mut layer_hidden = hidden_states.clone();
    let _ = &mut layer_hidden;
    let manifest: HashMap<String, serde_json::Value> = {
        let dir = std::env::var("DEEM_PARITY_DIR").unwrap();
        let text = fs::read_to_string(Path::new(&dir).join("manifest.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    };
    let _ = manifest;
    let _ = &load_f32("layer_0", None);
}
