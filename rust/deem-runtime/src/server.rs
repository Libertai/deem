//! `/v1/systemone` HTTP server — wire-compatible with `serve/deem_server.py`.

use crate::format::{
    build_prompt, read_answers_with_temps, render_inline, render_text, Answer, Question,
    QuestionSet, MAX_OPTIONS,
};
use crate::readout::Readout;
use serde_json::{json, Value};
use std::io::Read;
use std::sync::Arc;

const MAX_LEVELS: usize = 10;
const MAX_BODY_BYTES: u64 = 8 * 1024 * 1024;
const MAX_QUESTIONS: usize = 64;

pub struct Calibration {
    per_primitive: std::collections::HashMap<String, f32>,
    per_dataset: std::collections::HashMap<String, f32>,
}

impl Calibration {
    pub fn load(path: &std::path::Path, key: Option<&str>) -> Self {
        let mut cal = Calibration {
            per_primitive: Default::default(),
            per_dataset: Default::default(),
        };
        let Ok(text) = std::fs::read_to_string(path) else {
            return cal;
        };
        let Ok(parsed) = serde_json::from_str::<Value>(&text) else {
            return cal;
        };
        let root = match key {
            Some(k) => parsed.get(k).unwrap_or(&parsed),
            None => &parsed,
        };
        if let Some(pp) = root.get("per_primitive").and_then(|v| v.as_object()) {
            for (k, v) in pp {
                if let Some(f) = v.as_f64() {
                    cal.per_primitive.insert(k.clone(), f as f32);
                }
            }
        }
        if let Some(pd) = root.get("per_dataset").and_then(|v| v.as_object()) {
            for (k, v) in pd {
                if let Some(f) = v.as_f64() {
                    cal.per_dataset.insert(k.clone(), f as f32);
                }
            }
        }
        if cal.per_primitive.is_empty() {
            for (k, v) in parsed.as_object().into_iter().flatten() {
                if let Some(f) = v.as_f64() {
                    cal.per_primitive.insert(k.clone(), f as f32);
                }
            }
        }
        cal
    }

    fn temperature(&self, question_type: &str, dataset: Option<&str>) -> f32 {
        if let Some(d) = dataset {
            if let Some(t) = self.per_dataset.get(d) {
                return *t;
            }
        }
        self.per_primitive
            .get(question_type)
            .cloned()
            .unwrap_or(1.0)
    }
}

struct ParsedQuestion {
    question: Question,
    dataset: Option<String>,
    qid: String,
    qtype: String,
}

/// A request validation failure; `loc` is the path to the offending field,
/// starting with "body".
#[derive(Debug)]
struct Invalid {
    loc: Vec<String>,
    msg: String,
}

fn invalid(loc: &[&str], msg: impl Into<String>) -> Invalid {
    let mut path = vec!["body".to_string()];
    path.extend(loc.iter().map(|s| s.to_string()));
    Invalid {
        loc: path,
        msg: msg.into(),
    }
}

fn parse_request(body: &Value) -> Result<(Value, Vec<ParsedQuestion>), Invalid> {
    let state = body
        .get("state")
        .ok_or_else(|| invalid(&["state"], "field required"))?
        .clone();
    let questions_value = body
        .get("questions")
        .ok_or_else(|| invalid(&["questions"], "field required"))?;
    // Both wire forms, matching parse_questions in serve/deem_server.py:
    let questions: Vec<(String, &Value)> = if let Some(obj) = questions_value.as_object() {
        if obj.is_empty() {
            return Err(invalid(
                &["questions"],
                "must contain at least one question",
            ));
        }
        obj.iter().map(|(k, v)| (k.clone(), v)).collect()
    } else if let Some(arr) = questions_value.as_array() {
        let mut items = Vec::with_capacity(arr.len());
        for q in arr {
            let qid = q
                .get("id")
                .or_else(|| q.get("qid"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| invalid(&["questions"], "list-form questions need an 'id' field"))?;
            items.push((qid.to_string(), q));
        }
        items
    } else {
        return Err(invalid(&["questions"], "must be an object or a list"));
    };
    if questions.len() > MAX_QUESTIONS {
        return Err(invalid(
            &["questions"],
            format!("at most {MAX_QUESTIONS} questions per request"),
        ));
    }

    let top_dataset = body
        .get("dataset")
        .and_then(|d| d.as_str())
        .map(|s| s.to_string());

    let mut parsed = Vec::new();
    for (qid, q) in questions {
        let at = |field: &str, msg: String| invalid(&["questions", &qid, field], msg);
        let qtype = q
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| at("type", "field required".into()))?
            .to_string();
        let instructions = match q.get("instructions") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(v @ (Value::Object(_) | Value::Array(_))) => render_text(v),
            Some(_) => {
                return Err(at(
                    "instructions",
                    "must be a string, object or array".into(),
                ))
            }
        };
        let dataset = q
            .get("dataset")
            .and_then(|d| d.as_str())
            .or(top_dataset.as_deref())
            .map(|s| s.to_string());
        let criteria = q.get("criteria").filter(|c| !c.is_null());

        let question = match qtype.as_str() {
            "choice" => {
                let (options, descriptions): (Vec<String>, Vec<Option<String>>) =
                    match (criteria, q.get("options")) {
                        (Some(Value::Object(map)), _) => map
                            .iter()
                            .map(|(k, v)| (k.clone(), (!v.is_null()).then(|| render_inline(v))))
                            .unzip(),
                        (Some(_), _) => {
                            return Err(at(
                                "criteria",
                                "must map each option to a description".into(),
                            ))
                        }
                        (None, Some(Value::Array(items))) => {
                            let options = string_list(items).map_err(|m| at("options", m))?;
                            let descriptions = vec![None; options.len()];
                            (options, descriptions)
                        }
                        (None, _) => return Err(at("criteria", "field required".into())),
                    };
                check_labels(&options, 2, MAX_OPTIONS).map_err(|m| at("criteria", m))?;
                Question::Choice {
                    instructions,
                    options,
                    descriptions,
                }
            }
            "score" => {
                let levels: Vec<String> = match (criteria, q.get("levels")) {
                    (Some(Value::Array(items)), _) => items.iter().map(render_inline).collect(),
                    (Some(_), _) => return Err(at("criteria", "must be a list of levels".into())),
                    (None, Some(Value::Array(items))) => {
                        string_list(items).map_err(|m| at("levels", m))?
                    }
                    (None, _) => return Err(at("criteria", "field required".into())),
                };
                check_labels(&levels, 2, MAX_LEVELS).map_err(|m| at("criteria", m))?;
                Question::Score {
                    instructions,
                    levels,
                }
            }
            "noul" => {
                let (if_true, if_false) = match criteria {
                    None => (None, None),
                    Some(Value::Object(map)) => {
                        let side =
                            |key: &str| map.get(key).filter(|v| !v.is_null()).map(render_inline);
                        (side("true"), side("false"))
                    }
                    Some(_) => {
                        return Err(at(
                            "criteria",
                            "must be an object with 'true' and 'false'".into(),
                        ))
                    }
                };
                Question::Noul {
                    instructions,
                    if_true,
                    if_false,
                }
            }
            other => return Err(at("type", format!("unknown question type {other:?}"))),
        };
        parsed.push(ParsedQuestion {
            question,
            dataset,
            qid,
            qtype,
        });
    }
    Ok((state, parsed))
}

fn string_list(items: &[Value]) -> Result<Vec<String>, String> {
    items
        .iter()
        .map(|x| {
            x.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| "entries must be strings".to_string())
        })
        .collect()
}

fn check_labels(labels: &[String], min: usize, max: usize) -> Result<(), String> {
    if labels.len() < min || labels.len() > max {
        return Err(format!("needs {min}-{max} entries, got {}", labels.len()));
    }
    let mut seen = std::collections::HashSet::new();
    for label in labels {
        if label.trim().is_empty() {
            return Err("entries must be non-empty".to_string());
        }
        if label.contains(['\r', '\n']) {
            return Err("entries must not contain newlines".to_string());
        }
        if !seen.insert(label) {
            return Err(format!("duplicate entry {label:?}"));
        }
    }
    Ok(())
}

pub fn run(
    readout: Arc<Readout>,
    host: &str,
    port: u16,
    model_id: &str,
    calibration: Option<Calibration>,
) -> ! {
    use tiny_http::Server;
    let addr = format!("{host}:{port}");
    let server = Server::http(&addr).expect("bind failed");
    let model_id = Arc::new(model_id.to_string());
    let cal = Arc::new(calibration);
    eprintln!("[deem-rust] serving {addr} model={model_id}");
    for mut request in server.incoming_requests() {
        let readout = readout.clone();
        let model_id = model_id.clone();
        let cal = cal.clone();
        std::thread::spawn(move || {
            let response = handle_request(&readout, &mut request, &model_id, &cal);
            request.respond(response).ok();
        });
    }
    unreachable!()
}

type HttpResponse = tiny_http::Response<std::io::Cursor<Vec<u8>>>;

fn json_response(body: &Value, code: u16) -> HttpResponse {
    let header = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
        .expect("static header");
    tiny_http::Response::from_string(serde_json::to_string(body).unwrap())
        .with_status_code(code)
        .with_header(header)
}

/// TypeSafe error body for non-validation failures.
fn error_response(message: &str, error_type: &str, code: u16) -> HttpResponse {
    json_response(
        &json!({"detail": {"error_type": error_type, "message": message}}),
        code,
    )
}

/// TypeSafe (FastAPI-style) validation error body, status 422.
fn validation_response(err: &Invalid, error_type: &str) -> HttpResponse {
    json_response(
        &json!({"detail": [{"loc": err.loc, "msg": err.msg, "type": error_type}]}),
        422,
    )
}

fn handle_request(
    readout: &Arc<Readout>,
    request: &mut tiny_http::Request,
    model_id: &str,
    calibration: &Option<Calibration>,
) -> HttpResponse {
    let method = request.method().clone();
    let url = request.url().to_string();

    if method == tiny_http::Method::Get && url.starts_with("/v1/models") {
        return json_response(
            &json!({"models": [{"name": model_id,
                                "description": "Deem typed decision model",
                                "release_date": ""}]}),
            200,
        );
    }
    if method == tiny_http::Method::Get && url.starts_with("/health") {
        return json_response(
            &json!({"status": "ok", "model": model_id, "backend": "rust"}),
            200,
        );
    }
    if method == tiny_http::Method::Post && url.starts_with("/v1/systemone") {
        if request
            .body_length()
            .is_some_and(|n| n as u64 > MAX_BODY_BYTES)
        {
            return error_response("request body too large", "request_too_large", 413);
        }
        let mut body = Vec::new();
        let mut reader = request.as_reader().take(MAX_BODY_BYTES + 1);
        if reader.read_to_end(&mut body).is_err() {
            return error_response("could not read request body", "invalid_request_error", 400);
        }
        if body.len() as u64 > MAX_BODY_BYTES {
            return error_response("request body too large", "request_too_large", 413);
        }
        let payload: Value = match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => {
                let err = invalid(&[], format!("invalid JSON: {e}"));
                return validation_response(&err, "json_invalid");
            }
        };
        match complete(readout, &payload, model_id, calibration) {
            Ok(value) => json_response(&value, 200),
            Err(err) => validation_response(&err, "value_error"),
        }
    } else {
        error_response("not found", "not_found", 404)
    }
}

fn complete(
    readout: &Arc<Readout>,
    payload: &Value,
    model_id: &str,
    calibration: &Option<Calibration>,
) -> Result<Value, Invalid> {
    let (state, questions) = parse_request(payload)?;
    // The readout scores single-token letters only; more options than
    // letters would silently drop the tail.
    let max_letters = readout.tokenizer.letter_ids.len();
    for q in &questions {
        if q.question.n_valid() > max_letters {
            return Err(invalid(
                &["questions", &q.qid, "criteria"],
                format!("this model supports at most {max_letters} options"),
            ));
        }
    }

    // per-question isolation: one row per question
    let mut slot_logits = Vec::with_capacity(questions.len());
    let mut total_tokens = 0usize;
    for q in &questions {
        let mut qs = QuestionSet::default();
        qs.push(q.qid.clone(), q.question.clone());
        let prompt = build_prompt(&state, &qs, &Default::default());
        let (logits, tokens) = readout.slot_logits(&prompt);
        total_tokens += tokens;
        slot_logits.push(logits);
    }

    let default_cal = Calibration {
        per_primitive: Default::default(),
        per_dataset: Default::default(),
    };
    let cal = calibration.as_ref().unwrap_or(&default_cal);
    let temperatures: Vec<f32> = questions
        .iter()
        .map(|q| cal.temperature(&q.qtype, q.dataset.as_deref()))
        .collect();

    let question_set = {
        let mut qs = QuestionSet::default();
        for q in &questions {
            qs.push(q.qid.clone(), q.question.clone());
        }
        qs
    };
    let answers = read_answers_with_temps(
        &question_set,
        &slot_logits,
        &Default::default(),
        &temperatures,
    );

    let mut out = serde_json::Map::new();
    for (i, (qid, ans)) in answers.iter().enumerate() {
        out.insert(
            qid.clone(),
            answer_json(ans, &questions[i].question, temperatures[i]),
        );
    }

    Ok(json!({
        "model": model_id,
        "answers": out,
        "usage": {"input_tokens": total_tokens, "output_tokens": 0},
    }))
}

/// TypeSafe answer shape; fields outside the spec carry an `x_` prefix.
fn answer_json(ans: &Answer, question: &Question, temperature: f32) -> Value {
    match ans {
        Answer::Choice {
            choice,
            probabilities,
            confidence,
        } => {
            let probs: serde_json::Map<String, Value> = probabilities
                .iter()
                .map(|(k, v)| (k.clone(), json!(round5(*v))))
                .collect();
            json!({
                "type": "choice", "choice": choice,
                "probabilities": probs,
                "confidence": round5(*confidence),
                "x_temperature": temperature,
            })
        }
        Answer::Score {
            probabilities,
            expected,
            confidence,
            ..
        } => {
            let levels = question.labels().unwrap_or_default();
            let legend: serde_json::Map<String, Value> = levels
                .iter()
                .enumerate()
                .map(|(i, l)| (i.to_string(), json!(l)))
                .collect();
            let probs: serde_json::Map<String, Value> = probabilities
                .iter()
                .enumerate()
                .map(|(i, (_, v))| (i.to_string(), json!(round5(*v))))
                .collect();
            json!({
                "type": "score",
                "score": round5(*expected),
                "legend": legend,
                "probabilities": probs,
                "confidence": round5(*confidence),
                "x_temperature": temperature,
            })
        }
        Answer::Noul { value, confidence } => json!({
            "type": "noul",
            "noul": round5(*value),
            "x_confidence": round5(*confidence),
            "x_temperature": temperature,
        }),
    }
}

fn round5(v: f32) -> f64 {
    (v as f64 * 100000.0).round() / 100000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_request_object_form() {
        let body = json!({
            "state": "s",
            "questions": {
                "q1": {"type": "choice", "instructions": "pick",
                        "options": ["a", "b"]},
            },
        });
        let (_, questions) = parse_request(&body).unwrap();
        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].qid, "q1");
    }

    #[test]
    fn parse_request_list_form() {
        // The form used by the README quickstart and serve/deem_server.py.
        let body = json!({
            "state": "s",
            "questions": [
                {"id": "q1", "type": "noul", "instructions": "is it?"},
                {"qid": "q2", "type": "score", "instructions": "how much?",
                 "levels": ["low", "high"]},
            ],
        });
        let (_, questions) = parse_request(&body).unwrap();
        assert_eq!(questions.len(), 2);
        assert_eq!(questions[0].qid, "q1");
        assert_eq!(questions[0].qtype, "noul");
        assert_eq!(questions[1].qid, "q2");
        assert_eq!(questions[1].qtype, "score");
    }

    #[test]
    fn parse_request_list_form_requires_id() {
        let body = json!({
            "state": "s",
            "questions": [{"type": "noul", "instructions": "is it?"}],
        });
        assert!(parse_request(&body).is_err());
    }

    #[test]
    fn parse_request_rejects_scalar_questions() {
        let body = json!({"state": "s", "questions": "all of them"});
        assert!(parse_request(&body).is_err());
    }

    #[test]
    fn parse_request_typesafe_criteria() {
        let body: Value = serde_json::from_str(
            r#"{
                "model": "jev-latest",
                "state": "Help! My payouts have been failing for 3 days.",
                "questions": {
                    "urgent": {"type": "noul", "instructions": "Urgent?",
                               "criteria": {"true": "Time-sensitive", "false": "No urgency"}},
                    "team": {"type": "choice", "instructions": {"question": "Which team?"},
                             "criteria": {"technical": "Bugs", "billing": null, "sales": {"x": 1}}},
                    "mood": {"type": "score", "instructions": "Mood?",
                             "criteria": ["Calm", "Frustrated", "Very angry"]}
                }
            }"#,
        )
        .unwrap();
        let (_, questions) = parse_request(&body).unwrap();
        let ids: Vec<&str> = questions.iter().map(|q| q.qid.as_str()).collect();
        assert_eq!(ids, ["urgent", "team", "mood"]);
        match &questions[0].question {
            Question::Noul {
                if_true, if_false, ..
            } => {
                assert_eq!(if_true.as_deref(), Some("Time-sensitive"));
                assert_eq!(if_false.as_deref(), Some("No urgency"));
            }
            other => panic!("expected noul, got {other:?}"),
        }
        match &questions[1].question {
            Question::Choice {
                instructions,
                options,
                descriptions,
            } => {
                assert_eq!(instructions, r#"{"question":"Which team?"}"#);
                assert_eq!(options, &["technical", "billing", "sales"]);
                assert_eq!(
                    descriptions,
                    &[
                        Some("Bugs".to_string()),
                        None,
                        Some(r#"{"x":1}"#.to_string())
                    ]
                );
            }
            other => panic!("expected choice, got {other:?}"),
        }
        match &questions[2].question {
            Question::Score { levels, .. } => {
                assert_eq!(levels, &["Calm", "Frustrated", "Very angry"])
            }
            other => panic!("expected score, got {other:?}"),
        }
    }

    #[test]
    fn parse_request_rejects_bad_criteria() {
        for criteria in [json!(["a", "b"]), json!({"only": null})] {
            let body = json!({"state": "s", "questions": {
                "q": {"type": "choice", "instructions": "pick", "criteria": criteria}}});
            assert!(parse_request(&body).is_err());
        }
        let body = json!({"state": "s", "questions": {
            "q": {"type": "score", "instructions": "rate", "criteria": ["a", "a"]}}});
        assert!(parse_request(&body).is_err());
        let body = json!({"state": "s", "questions": {"q": {"type": "boolean"}}});
        let Err(err) = parse_request(&body) else {
            panic!("unknown type accepted")
        };
        assert_eq!(err.loc, ["body", "questions", "q", "type"]);
    }

    #[test]
    fn parse_request_noul_without_instructions() {
        let body = json!({"state": "s", "questions": {
            "q": {"type": "noul", "criteria": {"true": "yes", "false": "no"}}}});
        let (_, questions) = parse_request(&body).unwrap();
        assert!(matches!(&questions[0].question,
            Question::Noul { instructions, .. } if instructions.is_empty()));
    }

    #[test]
    fn answer_json_typesafe_shapes() {
        let score_q = Question::Score {
            instructions: "Mood?".into(),
            levels: vec!["Calm".into(), "Angry".into()],
        };
        let score = answer_json(
            &Answer::Score {
                level: "Angry".into(),
                probabilities: vec![("Calm".into(), 0.25), ("Angry".into(), 0.75)],
                expected: 0.75,
                confidence: 0.5,
            },
            &score_q,
            1.0,
        );
        assert_eq!(score["score"], json!(0.75));
        assert_eq!(score["legend"], json!({"0": "Calm", "1": "Angry"}));
        assert_eq!(score["probabilities"], json!({"0": 0.25, "1": 0.75}));

        let noul_q = Question::Noul {
            instructions: "?".into(),
            if_true: None,
            if_false: None,
        };
        let noul = answer_json(
            &Answer::Noul {
                value: 0.9,
                confidence: 0.8,
            },
            &noul_q,
            1.0,
        );
        assert_eq!(noul["type"], json!("noul"));
        assert_eq!(noul["noul"], json!(0.9));
    }
}
